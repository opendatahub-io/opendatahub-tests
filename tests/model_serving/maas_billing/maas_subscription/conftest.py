import secrets
import string
from collections.abc import Generator
from contextlib import ExitStack
from typing import Any

import pytest
import requests
from kubernetes.dynamic import DynamicClient
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.deployment import Deployment
from ocp_resources.llm_inference_service import LLMInferenceService
from ocp_resources.maas_auth_policy import MaaSAuthPolicy
from ocp_resources.maas_model_ref import MaaSModelRef
from ocp_resources.maas_subscription import MaaSSubscription
from ocp_resources.namespace import Namespace
from ocp_resources.resource import ResourceEditor
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.service_account import ServiceAccount
from pytest_testconfig import config as py_config
from simple_logger.logger import get_logger

from tests.model_serving.maas_billing.maas_subscription.utils import (
    MAAS_DB_NAMESPACE,
    MAAS_SUBSCRIPTION_NAMESPACE,
    create_api_key,
    create_maas_subscription,
    get_maas_postgres_resources,
    patch_llmisvc_with_maas_router_and_tiers,
    wait_for_postgres_connection_log,
    wait_for_postgres_deployment_ready,
)
from tests.model_serving.maas_billing.utils import build_maas_headers
from utilities.constants import DscComponents
from utilities.general import generate_random_name
from utilities.infra import create_inference_token, create_ns, login_with_user_password
from utilities.llmd_constants import ContainerImages, ModelStorage
from utilities.llmd_utils import create_llmisvc
from utilities.plugins.constant import OpenAIEnpoints

LOGGER = get_logger(name=__name__)

CHAT_COMPLETIONS = OpenAIEnpoints.CHAT_COMPLETIONS


@pytest.fixture(scope="session")
def maas_subscription_controller_enabled_latest(
    dsc_resource: DataScienceCluster,
    maas_postgres_prereqs: None,
    maas_gateway_api: None,
) -> Generator[DataScienceCluster, Any, Any]:
    """
    ensures postgres prerequisites exist before MaaS is switched to Managed.
    """
    component_patch = {
        DscComponents.KSERVE: {"modelsAsService": {"managementState": DscComponents.ManagementState.MANAGED}}
    }

    with ResourceEditor(patches={dsc_resource: {"spec": {"components": component_patch}}}):
        dsc_resource.wait_for_condition(
            condition="ModelsAsServiceReady",
            status="True",
            timeout=900,
        )
        dsc_resource.wait_for_condition(
            condition="Ready",
            status="True",
            timeout=600,
        )
        yield dsc_resource

    dsc_resource.wait_for_condition(condition="Ready", status="True", timeout=600)


@pytest.fixture(scope="class")
def maas_inference_service_tinyllama_free(
    admin_client: DynamicClient,
    maas_unprivileged_model_namespace: Namespace,
    maas_model_service_account: ServiceAccount,
    maas_gateway_api: None,
) -> Generator[LLMInferenceService, Any, Any]:
    with (
        create_llmisvc(
            client=admin_client,
            name="llm-s3-tinyllama-free",
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
        patch_llmisvc_with_maas_router_and_tiers(llm_service=llm_service, tiers=[]),
    ):
        llm_service.wait_for_condition(condition="Ready", status="True", timeout=900)
        yield llm_service


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
def maas_model_tinyllama_free(
    admin_client: DynamicClient,
    maas_inference_service_tinyllama_free: LLMInferenceService,
) -> Generator[MaaSModelRef]:

    with MaaSModelRef(
        client=admin_client,
        name=maas_inference_service_tinyllama_free.name,
        namespace=maas_inference_service_tinyllama_free.namespace,
        model_ref={
            "name": maas_inference_service_tinyllama_free.name,
            "namespace": maas_inference_service_tinyllama_free.namespace,
            "kind": "LLMInferenceService",
        },
        teardown=True,
        wait_for_resource=True,
    ) as maas_model:
        yield maas_model


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
def maas_auth_policy_tinyllama_free(
    admin_client: DynamicClient,
    maas_free_group: str,
    maas_model_tinyllama_free: MaaSModelRef,
    maas_subscription_namespace: Namespace,
) -> Generator[MaaSAuthPolicy]:

    with MaaSAuthPolicy(
        client=admin_client,
        name="tinyllama-free-access",
        namespace=maas_subscription_namespace.name,
        model_refs=[
            {
                "name": maas_model_tinyllama_free.name,
                "namespace": maas_model_tinyllama_free.namespace,
            }
        ],
        subjects={
            "groups": [
                {"name": "system:authenticated"},
                {"name": maas_free_group},
            ],
        },
        teardown=True,
        wait_for_resource=True,
    ) as maas_auth_policy_free:
        yield maas_auth_policy_free


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
def maas_subscription_tinyllama_free(
    admin_client: DynamicClient,
    maas_free_group: str,
    maas_model_tinyllama_free: MaaSModelRef,
    maas_subscription_namespace: Namespace,
) -> Generator[MaaSSubscription]:

    with MaaSSubscription(
        client=admin_client,
        name="tinyllama-free-subscription",
        namespace=maas_subscription_namespace.name,
        owner={
            "groups": [{"name": maas_free_group}],
        },
        model_refs=[
            {
                "name": maas_model_tinyllama_free.name,
                "namespace": maas_model_tinyllama_free.namespace,
                "tokenRateLimits": [{"limit": 100, "window": "1m"}],
            }
        ],
        priority=0,
        teardown=True,
        wait_for_resource=True,
    ) as maas_subscription_free:
        maas_subscription_free.wait_for_condition(condition="Ready", status="True", timeout=300)
        yield maas_subscription_free


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


@pytest.fixture(scope="session")
def maas_postgres_credentials() -> dict[str, str]:
    alphabet = string.ascii_letters + string.digits

    postgres_user = f"maas-{generate_random_name()}"
    postgres_db = f"maas-{generate_random_name()}"
    postgres_password = "".join(secrets.choice(alphabet) for _ in range(32))

    LOGGER.info(
        f"Generated PostgreSQL test credentials: "
        f"user={postgres_user}, db={postgres_db}, password_length={len(postgres_password)}"
    )

    return {
        "postgres_user": postgres_user,
        "postgres_password": postgres_password,
        "postgres_db": postgres_db,
    }


@pytest.fixture(scope="session")
def maas_postgres_prereqs(
    admin_client: DynamicClient,
    maas_postgres_credentials: dict[str, str],
) -> Generator[dict[Any, Any], Any, Any]:
    """
    Prepare PostgreSQL resources required by maas-api before MaaS API key tests run.
    """
    resources = get_maas_postgres_resources(
        client=admin_client,
        namespace=MAAS_DB_NAMESPACE,
        teardown_resources=True,
        postgres_user=maas_postgres_credentials["postgres_user"],
        postgres_password=maas_postgres_credentials["postgres_password"],
        postgres_db=maas_postgres_credentials["postgres_db"],
    )

    resources_instances: dict[Any, Any] = {}

    with ExitStack() as stack:
        for kind_name in [Secret, Service, Deployment]:
            resources_instances[kind_name] = []

            for resource_obj in resources[kind_name]:
                resources_instances[kind_name].append(stack.enter_context(resource_obj))

        for deployment in resources_instances[Deployment]:
            deployment.wait_for_condition(condition="Available", status="True", timeout=180)

        wait_for_postgres_deployment_ready(
            admin_client=admin_client,
            namespace=MAAS_DB_NAMESPACE,
            timeout=180,
        )
        wait_for_postgres_connection_log(
            admin_client=admin_client,
            namespace=MAAS_DB_NAMESPACE,
            timeout=180,
        )

        yield resources_instances


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


@pytest.fixture(scope="session")
def maas_subscription_namespace(unprivileged_client, admin_client):
    with create_ns(
        name=MAAS_SUBSCRIPTION_NAMESPACE,
        unprivileged_client=unprivileged_client,
        admin_client=admin_client,
    ) as ns:
        yield ns


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
            priority=0,
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
