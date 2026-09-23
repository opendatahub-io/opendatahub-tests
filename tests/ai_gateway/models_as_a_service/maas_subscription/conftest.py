import os
from collections.abc import Generator
from contextlib import ExitStack
from typing import Any

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.maas_auth_policy import MaaSAuthPolicy
from ocp_resources.maas_model_ref import MaaSModelRef
from ocp_resources.maas_subscription import MaaSSubscription
from ocp_resources.namespace import Namespace
from ocp_resources.service_account import ServiceAccount
from pytest_testconfig import config as py_config

from tests.ai_gateway.models_as_a_service.maas_api_key.utils import (
    MAAS_GATEWAY_AUTH_POLICY_NAME,
    wait_for_auth_policy_accepted,
)
from tests.ai_gateway.models_as_a_service.maas_subscription.utils import (
    MaaSLlmdScenario,
    ModelIdentityCollisionNames,
    build_model_identity_collision_names,
    create_maas_subscription,
    patch_llmisvc_with_maas_router_and_tiers,
)
from tests.ai_gateway.models_as_a_service.utils import (
    build_maas_headers,
    create_api_key,
    revoke_api_key,
)
from tests.model_serving.model_server.llmd.llmd_configs import TinyLlamaS3GpuConfig  # noqa: NIT001
from utilities.constants import MAAS_GATEWAY_NAMESPACE, ModelStorage
from utilities.general import generate_random_name
from utilities.image_constants import SharedImages
from utilities.infra import (
    create_inference_token,
    create_ns,
    login_with_user_password,
    s3_endpoint_secret,
)
from utilities.llmd_utils import create_llmisvc, create_llmisvc_from_config
from utilities.logger import RedactedString
from utilities.plugins.constant import OpenAIEnpoints
from utilities.resources.llm_inference_service import LLMInferenceService

LOGGER = structlog.get_logger(name=__name__)

CHAT_COMPLETIONS = OpenAIEnpoints.CHAT_COMPLETIONS
MAAS_LLM_D_DEBUG_LABEL = "e2e.opendatahub.io/maas-llmd-debug-run"
MAAS_LLM_D_SCENARIO_TOKENS_PER_MINUTE = 5_000


@pytest.fixture(scope="function")
def model_identity_collision_names() -> ModelIdentityCollisionNames:
    """Yield unique resource names for a model-identity collision test run."""
    return build_model_identity_collision_names(suffix=generate_random_name())


@pytest.fixture(scope="class")
def maas_inference_service_tinyllama_premium(
    admin_client: DynamicClient,
    maas_unprivileged_model_namespace: Namespace,
    maas_model_service_account: ServiceAccount,
    maas_gateway_api: None,
) -> Generator[LLMInferenceService, Any, Any]:
    with (
        create_llmisvc(
            client=admin_client,
            name="llm-s3-tinyllama-premium",
            namespace=maas_unprivileged_model_namespace.name,
            storage_uri=ModelStorage.S3.TINYLLAMA,
            container_image=SharedImages.VLLM_CPU,
            container_resources={
                "limits": {"cpu": "2", "memory": "12Gi"},
                "requests": {"cpu": "1", "memory": "8Gi"},
            },
            service_account=maas_model_service_account.name,
            wait=False,
            timeout=900,
        ) as llm_service,
        patch_llmisvc_with_maas_router_and_tiers(llm_service=llm_service, tiers=["premium"]),
    ):
        llm_service.wait_for_condition(condition="Ready", status="True", timeout=900)
        yield llm_service


@pytest.fixture(scope="class")
def maas_inference_service_tinyllama_load_balanced(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    maas_gateway_api: None,
) -> Generator[LLMInferenceService, Any, Any]:
    """Create a production-shaped, two-GPU-replica LLM-d service for MaaS."""
    with ExitStack() as stack:
        namespace = stack.enter_context(
            cm=create_ns(
                admin_client=admin_client,
                name=generate_random_name(prefix="maas-llmd-lb"),
            )
        )
        s3_secret = stack.enter_context(
            cm=s3_endpoint_secret(
                client=admin_client,
                name="tinyllama-s3",
                namespace=namespace.name,
                aws_access_key=request.getfixturevalue("aws_access_key_id"),
                aws_secret_access_key=request.getfixturevalue("aws_secret_access_key"),
                aws_s3_region=request.getfixturevalue("models_s3_bucket_region"),
                aws_s3_bucket=request.getfixturevalue("models_s3_bucket_name"),
                aws_s3_endpoint=request.getfixturevalue("models_s3_bucket_endpoint"),
            )
        )
        service_account = stack.enter_context(
            cm=ServiceAccount(
                client=admin_client,
                name="tinyllama-s3",
                namespace=namespace.name,
                secrets=[{"name": s3_secret.name}],
            )
        )
        config_cls = TinyLlamaS3GpuConfig.with_overrides(
            name="llm-s3-tinyllama-load-balanced",
            replicas=2,
            min_total_gpus=2,
            enable_auth=True,
        ).build(client=admin_client)
        llm_service = stack.enter_context(
            cm=create_llmisvc_from_config(
                config_cls=config_cls,
                client=admin_client,
                namespace=namespace.name,
                service_account=service_account.name,
            )
        )
        stack.enter_context(cm=patch_llmisvc_with_maas_router_and_tiers(llm_service=llm_service, tiers=[]))
        llm_service.wait_for_condition(condition="Ready", status="True", timeout=900)
        yield llm_service


@pytest.fixture(scope="class")
def maas_llmd_scenario(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    maas_gateway_api: None,
    maas_subscription_namespace: Namespace,
) -> Generator[MaaSLlmdScenario, Any, Any]:
    """Deploy a configured LLM-d scenario and expose it through MaaS."""
    keep_resources = os.getenv("MAAS_LLM_D_KEEP_RESOURCES", "").lower() in {"1", "true", "yes"}
    # Class-scoped parametrization creates one scenario per class, but the same
    # configuration can be exercised by more than one MaaS test module. Keep
    # the Kubernetes resource names unique when xdist schedules those classes
    # concurrently, while leaving enough room for dependent resource suffixes.
    scenario_name = generate_random_name(prefix=f"maas-{request.param.name[:45]}")
    debug_labels = {MAAS_LLM_D_DEBUG_LABEL: scenario_name} if keep_resources else {}
    config_cls = request.param.with_overrides(name=scenario_name, enable_auth=True).build(client=admin_client)
    with ExitStack() as stack:
        scenario_namespace = stack.enter_context(
            cm=create_ns(
                admin_client=admin_client,
                name=generate_random_name(prefix="maas-llmd"),
                teardown=not keep_resources,
                labels=debug_labels,
            )
        )
        if keep_resources:
            LOGGER.warning(
                "MaaS LLM-d debug retention is enabled; scenario resources will remain after pytest exits: "
                f"namespace={scenario_namespace.name}"
            )
        # The MaaS Gateway authenticates through Kubernetes TokenReview with
        # the Kubernetes service audience. Reuse the namespace's built-in
        # default ServiceAccount and mint a token for that exact audience;
        # this does not create a user, group, OAuth provider, or ServiceAccount.
        gateway_service_account = ServiceAccount(
            client=admin_client,
            name="default",
            namespace=scenario_namespace.name,
        )
        gateway_service_account.wait(timeout=60)
        gateway_user = f"system:serviceaccount:{scenario_namespace.name}:{gateway_service_account.name}"
        gateway_token = RedactedString(
            value=gateway_service_account.create_service_account_token(
                audiences=["https://kubernetes.default.svc"],
                expiration_seconds=86400,
            ).status.token
        )
        service_account = None
        if config_cls.storage_uri.startswith("s3://"):
            s3_secret = stack.enter_context(
                cm=s3_endpoint_secret(
                    client=admin_client,
                    name=f"{scenario_name}-s3",
                    namespace=scenario_namespace.name,
                    aws_access_key=request.getfixturevalue("aws_access_key_id"),
                    aws_secret_access_key=request.getfixturevalue("aws_secret_access_key"),
                    aws_s3_region=request.getfixturevalue("models_s3_bucket_region"),
                    aws_s3_bucket=request.getfixturevalue("models_s3_bucket_name"),
                    aws_s3_endpoint=request.getfixturevalue("models_s3_bucket_endpoint"),
                    teardown=not keep_resources,
                )
            )
            service_account = stack.enter_context(
                cm=ServiceAccount(
                    client=admin_client,
                    name=f"{scenario_name}-sa",
                    namespace=scenario_namespace.name,
                    secrets=[{"name": s3_secret.name}],
                    teardown=not keep_resources,
                )
            ).name
        llm_service = stack.enter_context(
            cm=create_llmisvc_from_config(
                config_cls=config_cls,
                client=admin_client,
                namespace=scenario_namespace.name,
                service_account=service_account,
                teardown=not keep_resources,
            )
        )
        stack.enter_context(
            cm=patch_llmisvc_with_maas_router_and_tiers(
                llm_service=llm_service,
                tiers=[],
                restore_on_exit=not keep_resources,
                labels=debug_labels,
            )
        )
        llm_service.wait_for_condition(condition="Ready", status="True", timeout=900)

        maas_model = stack.enter_context(
            cm=MaaSModelRef(
                client=admin_client,
                name=llm_service.name,
                namespace=llm_service.namespace,
                model_ref={
                    "name": llm_service.name,
                    "namespace": llm_service.namespace,
                    "kind": "LLMInferenceService",
                },
                label=debug_labels,
                teardown=not keep_resources,
                wait_for_resource=True,
            )
        )
        auth_policy = stack.enter_context(
            cm=MaaSAuthPolicy(
                client=admin_client,
                name=f"{llm_service.name}-access",
                namespace=maas_subscription_namespace.name,
                model_refs=[{"name": maas_model.name, "namespace": maas_model.namespace}],
                subjects={"users": [gateway_user]},
                label=debug_labels,
                teardown=not keep_resources,
                wait_for_resource=True,
            )
        )
        auth_policy.wait_for_condition(condition="Ready", status="True", timeout=300)
        subscription = stack.enter_context(
            cm=MaaSSubscription(
                client=admin_client,
                name=f"{llm_service.name}-subscription",
                namespace=maas_subscription_namespace.name,
                owner={"users": [gateway_user]},
                model_refs=[
                    {
                        "name": maas_model.name,
                        "namespace": maas_model.namespace,
                        # Prefix-cache scenarios send twelve long prompts and allow retries.
                        # Keep this above their worst-case request budget so MaaS rate
                        # limiting does not mask an EPP-routing regression.
                        "tokenRateLimits": [{"limit": MAAS_LLM_D_SCENARIO_TOKENS_PER_MINUTE, "window": "1m"}],
                    }
                ],
                priority=0,
                label=debug_labels,
                teardown=not keep_resources,
                wait_for_resource=True,
            )
        )
        subscription.wait_for_condition(condition="Ready", status="True", timeout=300)
        if keep_resources:
            LOGGER.warning(
                "Retained MaaS LLM-d resources: "
                f"namespace={scenario_namespace.name}, llmisvc={llm_service.name}, "
                f"model_ref={maas_model.name}, auth_policy={auth_policy.name}, "
                f"subscription={subscription.name}, debug_label={MAAS_LLM_D_DEBUG_LABEL}={scenario_name}"
            )
        wait_for_auth_policy_accepted(
            admin_client=admin_client,
            policy_name=MAAS_GATEWAY_AUTH_POLICY_NAME,
            namespace=MAAS_GATEWAY_NAMESPACE,
        )
        yield MaaSLlmdScenario(
            llmisvc=llm_service,
            subscription=subscription,
            gateway_token=gateway_token,
        )


@pytest.fixture(scope="class")
def model_url_llmd_scenario(
    maas_host: str,
) -> str:
    """Return MaaS's body-routed chat-completions URL for an LLM-d scenario."""
    return f"https://{maas_host}{CHAT_COMPLETIONS}"


@pytest.fixture(scope="function")
def api_key_bound_to_llmd_scenario(
    base_url: str,
    maas_llmd_scenario: MaaSLlmdScenario,
    request_session_http: requests.Session,
) -> Generator[str, Any, Any]:
    """Create and revoke an API key for the current LLM-d MaaS scenario."""
    _, body = create_api_key(
        base_url=base_url,
        ocp_user_token=maas_llmd_scenario.gateway_token,
        request_session_http=request_session_http,
        api_key_name=f"e2e-maas-{maas_llmd_scenario.llmisvc.name}",
        subscription=maas_llmd_scenario.subscription.name,
    )
    try:
        yield body["key"]
    finally:
        revoke_api_key(
            request_session_http=request_session_http,
            base_url=base_url,
            key_id=body["id"],
            ocp_user_token=maas_llmd_scenario.gateway_token,
        )


@pytest.fixture(scope="class")
def maas_model_tinyllama_load_balanced(
    admin_client: DynamicClient,
    maas_inference_service_tinyllama_load_balanced: LLMInferenceService,
) -> Generator[MaaSModelRef, Any, Any]:
    """Register the load-balanced LLM-d service with MaaS."""
    with MaaSModelRef(
        client=admin_client,
        name=maas_inference_service_tinyllama_load_balanced.name,
        namespace=maas_inference_service_tinyllama_load_balanced.namespace,
        model_ref={
            "name": maas_inference_service_tinyllama_load_balanced.name,
            "namespace": maas_inference_service_tinyllama_load_balanced.namespace,
            "kind": "LLMInferenceService",
        },
        teardown=True,
        wait_for_resource=True,
    ) as maas_model:
        yield maas_model


@pytest.fixture(scope="class")
def maas_auth_policy_tinyllama_load_balanced(
    admin_client: DynamicClient,
    maas_free_group: str,
    maas_model_tinyllama_load_balanced: MaaSModelRef,
    maas_subscription_namespace: Namespace,
) -> Generator[MaaSAuthPolicy, Any, Any]:
    """Grant the free-tier user access to the load-balanced LLM-d model."""
    with MaaSAuthPolicy(
        client=admin_client,
        name="tinyllama-load-balanced-access",
        namespace=maas_subscription_namespace.name,
        model_refs=[
            {
                "name": maas_model_tinyllama_load_balanced.name,
                "namespace": maas_model_tinyllama_load_balanced.namespace,
            }
        ],
        subjects={"groups": [{"name": maas_free_group}]},
        teardown=True,
        wait_for_resource=True,
    ) as maas_auth_policy:
        yield maas_auth_policy


@pytest.fixture(scope="class")
def maas_subscription_tinyllama_load_balanced(
    admin_client: DynamicClient,
    maas_free_group: str,
    maas_model_tinyllama_load_balanced: MaaSModelRef,
    maas_subscription_namespace: Namespace,
) -> Generator[MaaSSubscription, Any, Any]:
    """Create a free-tier subscription for the load-balanced LLM-d model."""
    with MaaSSubscription(
        client=admin_client,
        name="tinyllama-load-balanced-subscription",
        namespace=maas_subscription_namespace.name,
        owner={"groups": [{"name": maas_free_group}]},
        model_refs=[
            {
                "name": maas_model_tinyllama_load_balanced.name,
                "namespace": maas_model_tinyllama_load_balanced.namespace,
                "tokenRateLimits": [{"limit": 1000, "window": "1m"}],
            }
        ],
        priority=0,
        teardown=True,
        wait_for_resource=True,
    ) as maas_subscription:
        maas_subscription.wait_for_condition(condition="Ready", status="True", timeout=300)
        yield maas_subscription


@pytest.fixture(scope="class")
def maas_model_tinyllama_premium(
    admin_client: DynamicClient,
    maas_inference_service_tinyllama_premium: LLMInferenceService,
) -> Generator[MaaSModelRef]:

    with MaaSModelRef(
        client=admin_client,
        name=maas_inference_service_tinyllama_premium.name,
        namespace=maas_inference_service_tinyllama_premium.namespace,
        model_ref={
            "name": maas_inference_service_tinyllama_premium.name,
            "namespace": maas_inference_service_tinyllama_premium.namespace,
            "kind": "LLMInferenceService",
        },
        teardown=True,
        wait_for_resource=True,
    ) as maas_model:
        yield maas_model


@pytest.fixture(scope="class")
def maas_auth_policy_tinyllama_premium(
    admin_client: DynamicClient,
    maas_premium_group: str,
    maas_model_tinyllama_premium: MaaSModelRef,
    maas_subscription_namespace: Namespace,
) -> Generator[MaaSAuthPolicy]:

    with MaaSAuthPolicy(
        client=admin_client,
        name="tinyllama-premium-access",
        namespace=maas_subscription_namespace.name,
        model_refs=[
            {
                "name": maas_model_tinyllama_premium.name,
                "namespace": maas_model_tinyllama_premium.namespace,
            }
        ],
        subjects={
            "groups": [{"name": maas_premium_group}],
        },
        teardown=True,
        wait_for_resource=True,
    ) as maas_auth_policy_premium:
        yield maas_auth_policy_premium


@pytest.fixture(scope="class")
def maas_subscription_tinyllama_premium(
    admin_client: DynamicClient,
    maas_premium_group: str,
    maas_model_tinyllama_premium: MaaSModelRef,
    maas_subscription_namespace: Namespace,
) -> Generator[MaaSSubscription]:

    with MaaSSubscription(
        client=admin_client,
        name="tinyllama-premium-subscription",
        namespace=maas_subscription_namespace.name,
        owner={
            "groups": [{"name": maas_premium_group}],
        },
        model_refs=[
            {
                "name": maas_model_tinyllama_premium.name,
                "namespace": maas_model_tinyllama_premium.namespace,
                "tokenRateLimits": [{"limit": 1000, "window": "1m"}],
            }
        ],
        priority=0,
        teardown=True,
        wait_for_resource=True,
    ) as maas_subscription_premium:
        maas_subscription_premium.wait_for_condition(condition="Ready", status="True", timeout=300)
        yield maas_subscription_premium


@pytest.fixture(scope="class")
def models_url(base_url: str) -> str:
    """GET /v1/models endpoint URL."""
    return f"{base_url}/v1/models"


@pytest.fixture(scope="class")
def model_url_tinyllama_free(
    maas_scheme: str,
    maas_host: str,
    maas_inference_service_tinyllama_free: LLMInferenceService,
) -> str:
    deployment_name = maas_inference_service_tinyllama_free.name
    url = f"{maas_scheme}://{maas_host}/llm/{deployment_name}{CHAT_COMPLETIONS}"
    LOGGER.info(f"MaaS: constructed model_url={url} (deployment={deployment_name})")
    return url


@pytest.fixture(scope="class")
def model_url_tinyllama_premium(
    maas_scheme: str,
    maas_host: str,
    maas_inference_service_tinyllama_premium: LLMInferenceService,
) -> str:
    deployment_name = maas_inference_service_tinyllama_premium.name
    url = f"{maas_scheme}://{maas_host}/llm/{deployment_name}{CHAT_COMPLETIONS}"
    LOGGER.info(f"MaaS: constructed model_url={url} (deployment={deployment_name})")
    return url


@pytest.fixture(scope="class")
def model_url_tinyllama_load_balanced(
    maas_host: str,
) -> str:
    """Return MaaS's body-routed chat-completions URL for the load-balanced LLM-d model."""
    url = f"https://{maas_host}{CHAT_COMPLETIONS}"
    LOGGER.info(f"MaaS: constructed body-routed load-balanced model URL={url}")
    return url


@pytest.fixture(scope="class")
def api_key_bound_to_load_balanced_subscription(
    request_session_http: requests.Session,
    base_url: str,
    ocp_token_for_actor: str,
    maas_subscription_tinyllama_load_balanced: MaaSSubscription,
) -> Generator[str, Any, Any]:
    """Create and revoke a free-user API key bound to the load-balanced model subscription."""
    _, body = create_api_key(
        base_url=base_url,
        ocp_user_token=ocp_token_for_actor,
        request_session_http=request_session_http,
        api_key_name=f"e2e-llmd-load-balanced-{generate_random_name()}",
        subscription=maas_subscription_tinyllama_load_balanced.name,
    )
    try:
        yield body["key"]
    finally:
        revoke_api_key(
            request_session_http=request_session_http,
            base_url=base_url,
            key_id=body["id"],
            ocp_user_token=ocp_token_for_actor,
        )


@pytest.fixture(scope="class")
def maas_api_key_for_actor(
    request_session_http: requests.Session,
    base_url: str,
    ocp_token_for_actor: str,
    maas_subscription_controller_enabled_latest: None,
    maas_gateway_api: None,
    maas_api_gateway_reachable: None,
) -> str:
    """
    Create an API key for the current actor (admin/free/premium).

    Flow:
    - Use OpenShift token (ocp_token_for_actor) to create an API key via MaaS API.
    - Use the plaintext API key for gateway inference: Authorization: Bearer <sk-...>.
    """
    api_key_name = f"odh-sub-tests-{generate_random_name()}"

    _, body = create_api_key(
        base_url=base_url,
        ocp_user_token=ocp_token_for_actor,
        request_session_http=request_session_http,
        api_key_name=api_key_name,
        request_timeout_seconds=60,
    )

    return body["key"]


@pytest.fixture(scope="class")
def maas_headers_for_actor_api_key(maas_api_key_for_actor: str) -> dict[str, str]:
    """
    Headers for gateway inference using API key (new implementation).
    """
    return build_maas_headers(token=maas_api_key_for_actor)


@pytest.fixture(scope="function")
def extra_subscription_with_api_key(
    request_session_http: requests.Session,
    base_url: str,
    ocp_token_for_actor: str,
    admin_client: DynamicClient,
    maas_free_group: str,
    maas_model_tinyllama_free: MaaSModelRef,
    maas_subscription_namespace: Namespace,
    maas_subscription_tinyllama_free: MaaSSubscription,
    maas_subscription_controller_enabled_latest: None,
    maas_gateway_api: None,
    maas_api_gateway_reachable: None,
) -> Generator[str, Any, Any]:
    """
    Creates an extra subscription (for nonexistent-group, priority=1) and an API key
    bound to the original free subscription. Verifies the user's key still works even
    with a second subscription present (OR-logic fix). Revokes key on teardown.
    """
    with create_maas_subscription(
        admin_client=admin_client,
        subscription_namespace=maas_subscription_namespace.name,
        subscription_name="extra-subscription",
        owner_group_name="nonexistent-group-xyz",
        model_name=maas_model_tinyllama_free.name,
        model_namespace=maas_model_tinyllama_free.namespace,
        tokens_per_minute=999,
        window="1m",
        priority=1,
        teardown=True,
        wait_for_resource=True,
    ) as extra_subscription:
        extra_subscription.wait_for_condition(condition="Ready", status="True", timeout=300)
        _, body = create_api_key(
            base_url=base_url,
            ocp_user_token=ocp_token_for_actor,
            request_session_http=request_session_http,
            api_key_name=f"e2e-one-of-two-{generate_random_name()}",
            subscription=maas_subscription_tinyllama_free.name,
        )
        yield body["key"]
        revoke_api_key(
            request_session_http=request_session_http,
            base_url=base_url,
            key_id=body["id"],
            ocp_user_token=ocp_token_for_actor,
        )


@pytest.fixture(scope="function")
def high_tier_subscription_with_api_key(
    request_session_http: requests.Session,
    base_url: str,
    ocp_token_for_actor: str,
    admin_client: DynamicClient,
    maas_free_group: str,
    maas_model_tinyllama_free: MaaSModelRef,
    maas_subscription_namespace: Namespace,
    maas_subscription_tinyllama_free: MaaSSubscription,
    maas_subscription_controller_enabled_latest: None,
    maas_gateway_api: None,
    maas_api_gateway_reachable: None,
) -> Generator[str, Any, Any]:
    """
    Creates a high-priority subscription (priority=10) for the free group and an API key
    bound to it. Returns the API key. Revokes key and cleans up subscription on teardown.
    """
    with create_maas_subscription(
        admin_client=admin_client,
        subscription_namespace=maas_subscription_namespace.name,
        subscription_name="high-tier-subscription",
        owner_group_name=maas_free_group,
        model_name=maas_model_tinyllama_free.name,
        model_namespace=maas_model_tinyllama_free.namespace,
        tokens_per_minute=9999,
        window="1m",
        priority=10,
        teardown=True,
        wait_for_resource=True,
    ) as high_tier_subscription:
        high_tier_subscription.wait_for_condition(condition="Ready", status="True", timeout=300)
        _, body = create_api_key(
            base_url=base_url,
            ocp_user_token=ocp_token_for_actor,
            request_session_http=request_session_http,
            api_key_name=f"e2e-high-tier-{generate_random_name()}",
            subscription=high_tier_subscription.name,
        )
        yield body["key"]
        revoke_api_key(
            request_session_http=request_session_http,
            base_url=base_url,
            key_id=body["id"],
            ocp_user_token=ocp_token_for_actor,
        )


@pytest.fixture(scope="function")
def api_key_bound_to_system_auth_subscription(
    request_session_http: requests.Session,
    base_url: str,
    ocp_token_for_actor: str,
    premium_system_authenticated_access: dict,
    maas_subscription_controller_enabled_latest: None,
    maas_gateway_api: None,
    maas_api_gateway_reachable: None,
) -> Generator[str, Any, Any]:
    """
    API key bound to the system:authenticated subscription on the premium model.
    Used for tests that verify OR-logic auth policy access. Revoked on teardown.
    """
    _, body = create_api_key(
        base_url=base_url,
        ocp_user_token=ocp_token_for_actor,
        request_session_http=request_session_http,
        api_key_name=f"e2e-system-auth-{generate_random_name()}",
        subscription=premium_system_authenticated_access["subscription"].name,
    )
    yield body["key"]
    revoke_api_key(
        request_session_http=request_session_http,
        base_url=base_url,
        key_id=body["id"],
        ocp_user_token=ocp_token_for_actor,
    )


@pytest.fixture(scope="class")
def api_key_bound_to_free_subscription(
    request_session_http: requests.Session,
    base_url: str,
    ocp_token_for_actor: str,
    maas_subscription_tinyllama_free: MaaSSubscription,
    maas_subscription_controller_enabled_latest: None,
    maas_gateway_api: None,
    maas_api_gateway_reachable: None,
) -> Generator[str, Any, Any]:
    """
    API key bound to the free subscription at mint time. Revoked on teardown.
    """
    _, body = create_api_key(
        base_url=base_url,
        ocp_user_token=ocp_token_for_actor,
        request_session_http=request_session_http,
        api_key_name=f"e2e-auth-enforce-{generate_random_name()}",
        subscription=maas_subscription_tinyllama_free.name,
    )
    yield body["key"]
    revoke_api_key(
        request_session_http=request_session_http,
        base_url=base_url,
        key_id=body["id"],
        ocp_user_token=ocp_token_for_actor,
    )


@pytest.fixture(scope="class")
def api_key_bound_to_premium_subscription(
    request_session_http: requests.Session,
    base_url: str,
    ocp_token_for_actor: str,
    maas_subscription_tinyllama_premium: MaaSSubscription,
    maas_subscription_controller_enabled_latest: None,
    maas_gateway_api: None,
    maas_api_gateway_reachable: None,
) -> Generator[str, Any, Any]:
    """
    API key bound to the premium subscription at mint time. Revoked on teardown.
    """
    _, body = create_api_key(
        base_url=base_url,
        ocp_user_token=ocp_token_for_actor,
        request_session_http=request_session_http,
        api_key_name=f"e2e-sub-enforce-{generate_random_name()}",
        subscription=maas_subscription_tinyllama_premium.name,
    )
    yield body["key"]
    revoke_api_key(
        request_session_http=request_session_http,
        base_url=base_url,
        key_id=body["id"],
        ocp_user_token=ocp_token_for_actor,
    )


@pytest.fixture(scope="function")
def deleted_sub_service_account(
    admin_client: DynamicClient,
    maas_api_server_url: str,
    original_user: str,
) -> Generator[str, Any, Any]:
    """Create a dedicated SA and return its token for deleted-subscription testing."""
    sa_name = f"e2e-deleted-sub-sa-{generate_random_name()}"
    applications_namespace = py_config["applications_namespace"]

    with ServiceAccount(
        client=admin_client,
        namespace=applications_namespace,
        name=sa_name,
        teardown=True,
    ) as sa:
        sa.wait(timeout=60)

        ok = login_with_user_password(api_address=maas_api_server_url, user=original_user)
        assert ok, f"Failed to login as original_user={original_user}"

        yield create_inference_token(model_service_account=sa)


@pytest.fixture(scope="function")
def api_key_for_deleted_subscription(
    request_session_http: requests.Session,
    base_url: str,
    admin_client: DynamicClient,
    deleted_sub_service_account: str,
    maas_model_tinyllama_free: MaaSModelRef,
    maas_subscription_tinyllama_free: MaaSSubscription,
) -> Generator[str, Any, Any]:
    """Create an API key bound to a temp subscription, then delete the subscription.

    Returns the plaintext API key whose subscription no longer exists.
    """
    temp_sub_name = f"e2e-deleted-sub-{generate_random_name()}"

    with create_maas_subscription(
        admin_client=admin_client,
        subscription_namespace=maas_subscription_tinyllama_free.namespace,
        subscription_name=temp_sub_name,
        owner_group_name="system:authenticated",
        model_name=maas_model_tinyllama_free.name,
        model_namespace=maas_model_tinyllama_free.namespace,
        tokens_per_minute=100,
        window="1m",
        priority=20,
        teardown=True,
        wait_for_resource=True,
    ) as temp_subscription:
        temp_subscription.wait_for_condition(condition="Ready", status="True", timeout=300)

        _, body = create_api_key(
            base_url=base_url,
            ocp_user_token=deleted_sub_service_account,
            request_session_http=request_session_http,
            api_key_name=f"e2e-deleted-sub-key-{generate_random_name()}",
            subscription=temp_sub_name,
        )
        api_key_plaintext = body["key"]
        LOGGER.info(f"api_key_for_deleted_subscription: created key id={body['id']} bound to '{temp_sub_name}'")

    LOGGER.info(f"api_key_for_deleted_subscription: subscription '{temp_sub_name}' deleted")
    yield api_key_plaintext

    revoke_api_key(
        request_session_http=request_session_http,
        base_url=base_url,
        key_id=body["id"],
        ocp_user_token=deleted_sub_service_account,
    )


@pytest.fixture(scope="class")
def exhausted_token_quota(
    request_session_http: requests.Session,
    model_url_tinyllama_free: str,
    api_key_bound_to_free_subscription: str,
    maas_subscription_tinyllama_free: MaaSSubscription,
) -> None:
    """Exhaust the free-tier token quota by sending inference requests until 429."""
    headers = build_maas_headers(token=api_key_bound_to_free_subscription)
    headers["x-maas-subscription"] = maas_subscription_tinyllama_free.name

    max_requests = 10
    for attempt in range(max_requests):
        response = request_session_http.post(
            url=model_url_tinyllama_free,
            headers=headers,
            json={
                "model": "llm-s3-tinyllama-free",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 50,
            },
            timeout=60,
        )
        if response.status_code == 429:
            LOGGER.info(f"[models] Rate limit hit after {attempt + 1} inference request(s)")
            return

        assert response.status_code == 200, (
            f"Unexpected status {response.status_code} during inference "
            f"(attempt {attempt + 1}): {(response.text or '')[:200]}"
        )

    pytest.fail(f"Could not exhaust token quota within {max_requests} requests")


@pytest.fixture(scope="class")
def maas_wrong_group_service_account_token(
    maas_api_server_url: str,
    original_user: str,
    admin_client: DynamicClient,
) -> Generator[str]:
    applications_namespace = py_config["applications_namespace"]

    with ServiceAccount(
        client=admin_client,
        namespace=applications_namespace,
        name="e2e-wrong-group-sa",
        teardown=True,
    ) as sa:
        sa.wait(timeout=60)

        ok = login_with_user_password(api_address=maas_api_server_url, user=original_user)
        assert ok, f"Failed to login as original_user={original_user}"

        raw_token = create_inference_token(model_service_account=sa)
        yield raw_token


@pytest.fixture(scope="class")
def maas_headers_for_wrong_group_sa(maas_wrong_group_service_account_token: str) -> dict:
    return build_maas_headers(token=maas_wrong_group_service_account_token)


@pytest.fixture(scope="function")
def temporary_system_authenticated_subscription(
    admin_client: DynamicClient,
    maas_subscription_tinyllama_free: MaaSSubscription,
    maas_model_tinyllama_free: MaaSModelRef,
) -> Generator[MaaSSubscription, Any, Any]:
    """
    Creates a temporary subscription owned by system:authenticated.
    Used for cascade deletion tests.
    """

    subscription_name = f"e2e-temp-sub-{generate_random_name()}"

    with create_maas_subscription(
        admin_client=admin_client,
        subscription_namespace=maas_subscription_tinyllama_free.namespace,
        subscription_name=subscription_name,
        owner_group_name="system:authenticated",
        model_name=maas_model_tinyllama_free.name,
        model_namespace=maas_model_tinyllama_free.namespace,
        tokens_per_minute=50,
        window="1m",
        priority=2,
        teardown=True,
        wait_for_resource=True,
    ) as temporary_subscription:
        temporary_subscription.wait_for_condition(
            condition="Ready",
            status="True",
            timeout=300,
        )

        LOGGER.info(
            f"Created temporary subscription {temporary_subscription.name} for model {maas_model_tinyllama_free.name}"
        )

        yield temporary_subscription

        LOGGER.info(f"Fixture teardown: ensuring subscription {temporary_subscription.name} is removed")
        temporary_subscription.clean_up(wait=True)


@pytest.fixture(scope="function")
def premium_system_authenticated_access(
    admin_client: DynamicClient,
    maas_model_tinyllama_premium: MaaSModelRef,
    maas_subscription_tinyllama_premium: MaaSSubscription,
) -> Generator[dict[str, Any], Any, Any]:
    """
    Creates an extra AuthPolicy and matching subscription for system:authenticated
    on the premium model.
    """

    auth_policy_name = f"e2e-premium-system-auth-{generate_random_name()}"
    subscription_name = f"e2e-premium-system-auth-sub-{generate_random_name()}"

    with (
        MaaSAuthPolicy(
            client=admin_client,
            name=auth_policy_name,
            namespace=maas_subscription_tinyllama_premium.namespace,
            model_refs=[
                {
                    "name": maas_model_tinyllama_premium.name,
                    "namespace": maas_model_tinyllama_premium.namespace,
                }
            ],
            subjects={"groups": [{"name": "system:authenticated"}]},
            teardown=False,
            wait_for_resource=True,
        ) as extra_auth_policy,
        create_maas_subscription(
            admin_client=admin_client,
            subscription_namespace=maas_subscription_tinyllama_premium.namespace,
            subscription_name=subscription_name,
            owner_group_name="system:authenticated",
            model_name=maas_model_tinyllama_premium.name,
            model_namespace=maas_model_tinyllama_premium.namespace,
            tokens_per_minute=100,
            window="1m",
            priority=1,
            teardown=True,
            wait_for_resource=True,
        ) as system_authenticated_subscription,
    ):
        extra_auth_policy.wait_for_condition(condition="Ready", status="True", timeout=300)
        system_authenticated_subscription.wait_for_condition(
            condition="Ready",
            status="True",
            timeout=300,
        )

        LOGGER.info(
            f"Created extra AuthPolicy {extra_auth_policy.name} and subscription "
            f"{system_authenticated_subscription.name} for premium model "
            f"{maas_model_tinyllama_premium.name}"
        )

        yield {
            "auth_policy": extra_auth_policy,
            "subscription": system_authenticated_subscription,
        }

        if extra_auth_policy.exists:
            LOGGER.info(f"Fixture teardown: ensuring AuthPolicy {extra_auth_policy.name} is removed")
            extra_auth_policy.clean_up(wait=True)


@pytest.fixture(scope="function")
def free_actor_premium_subscription(
    admin_client: DynamicClient,
    maas_model_tinyllama_premium: MaaSModelRef,
    maas_subscription_tinyllama_premium: MaaSSubscription,
) -> Generator[MaaSSubscription, Any, Any]:
    """
    Creates a subscription for system:authenticated on the premium model.
    Used to verify that having a subscription alone is not sufficient —
    the actor must also be listed in the model's MaaSAuthPolicy.
    """
    with create_maas_subscription(
        admin_client=admin_client,
        subscription_namespace=maas_subscription_tinyllama_premium.namespace,
        subscription_name="e2e-free-actor-premium-sub",
        owner_group_name="system:authenticated",
        model_name=maas_model_tinyllama_premium.name,
        model_namespace=maas_model_tinyllama_premium.namespace,
        tokens_per_minute=100,
        window="1m",
        priority=5,
        teardown=True,
        wait_for_resource=True,
    ) as sub_for_free_actor:
        sub_for_free_actor.wait_for_condition(condition="Ready", status="True", timeout=300)
        LOGGER.info(
            f"Created subscription {sub_for_free_actor.name} for system:authenticated "
            f"on premium model {maas_model_tinyllama_premium.name}"
        )
        yield sub_for_free_actor
