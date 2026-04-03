from collections.abc import Generator
from typing import Any

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.llm_inference_service import LLMInferenceService
from ocp_resources.maas_auth_policy import MaaSAuthPolicy
from ocp_resources.maas_model_ref import MaaSModelRef
from ocp_resources.maas_subscription import MaaSSubscription
from ocp_resources.namespace import Namespace
from ocp_resources.service_account import ServiceAccount
from pytest_testconfig import config as py_config

from tests.model_serving.maas_billing.maas_subscription.utils import (
    create_maas_subscription,
    patch_llmisvc_with_maas_router_and_tiers,
)
from tests.model_serving.maas_billing.utils import build_maas_headers, create_api_key, revoke_api_key
from utilities.general import generate_random_name
from utilities.infra import create_inference_token, login_with_user_password
from utilities.llmd_constants import ContainerImages, ModelStorage
from utilities.llmd_utils import create_llmisvc
from utilities.plugins.constant import OpenAIEnpoints

LOGGER = structlog.get_logger(name=__name__)

CHAT_COMPLETIONS = OpenAIEnpoints.CHAT_COMPLETIONS


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
            storage_uri=ModelStorage.TINYLLAMA_S3,
            container_image=ContainerImages.VLLM_CPU,
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
        priority=0,
        teardown=False,
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
def second_free_subscription(
    admin_client: DynamicClient,
    maas_free_group: str,
    maas_model_tinyllama_free: MaaSModelRef,
    maas_subscription_tinyllama_free: MaaSSubscription,
) -> Generator[MaaSSubscription, Any, Any]:
    """
    Creates a second subscription for maas_free_group on the free model.
    Used to simulate an ambiguous subscription selection (two qualifying subscriptions,
    no x-maas-subscription header).
    """
    with create_maas_subscription(
        admin_client=admin_client,
        subscription_namespace=maas_subscription_tinyllama_free.namespace,
        subscription_name="e2e-second-free-subscription",
        owner_group_name=maas_free_group,
        model_name=maas_model_tinyllama_free.name,
        model_namespace=maas_model_tinyllama_free.namespace,
        tokens_per_minute=500,
        window="1m",
        priority=5,
        teardown=True,
        wait_for_resource=True,
    ) as second_subscription:
        second_subscription.wait_for_condition(condition="Ready", status="True", timeout=300)
        LOGGER.info(
            f"Created second free subscription {second_subscription.name} for model {maas_model_tinyllama_free.name}"
        )
        yield second_subscription


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
        priority=0,
        teardown=True,
        wait_for_resource=True,
    ) as sub_for_free_actor:
        sub_for_free_actor.wait_for_condition(condition="Ready", status="True", timeout=300)
        LOGGER.info(
            f"Created subscription {sub_for_free_actor.name} for system:authenticated "
            f"on premium model {maas_model_tinyllama_premium.name}"
        )
        yield sub_for_free_actor
