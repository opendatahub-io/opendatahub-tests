from collections.abc import Generator
from typing import Any

import pytest
import requests
from kubernetes.dynamic import DynamicClient
from ocp_resources.llm_inference_service import LLMInferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.service_account import ServiceAccount
from pytest_testconfig import config as py_config
from simple_logger.logger import get_logger

from tests.model_serving.model_server.maas_billing.maas_subscription.utils import (
    create_api_key,
    patch_llmisvc_with_maas_router_and_tiers,
)
from tests.model_serving.model_server.maas_billing.utils import build_maas_headers
from utilities.general import generate_random_name
from utilities.infra import create_inference_token, login_with_user_password
from utilities.llmd_constants import ContainerImages, ModelStorage
from utilities.llmd_utils import create_llmisvc
from utilities.plugins.constant import OpenAIEnpoints
from utilities.resources.maa_s_auth_policy import MaaSAuthPolicy
from utilities.resources.maa_s_subscription import MaaSSubscription
from ocp_resources.maas_model_ref import MaaSModelRef

LOGGER = get_logger(name=__name__)

CHAT_COMPLETIONS = OpenAIEnpoints.CHAT_COMPLETIONS

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
    applications_namespace = py_config["applications_namespace"]

    with MaaSModelRef(
        client=admin_client,
        name=maas_inference_service_tinyllama_free.name,
        namespace=applications_namespace,
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
    applications_namespace = py_config["applications_namespace"]

    with MaaSModelRef(
        client=admin_client,
        name=maas_inference_service_tinyllama_premium.name,
        namespace=applications_namespace,
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
) -> Generator[MaaSAuthPolicy]:
    applications_namespace = py_config["applications_namespace"]

    with MaaSAuthPolicy(
        client=admin_client,
        name="tinyllama-free-access",
        namespace=applications_namespace,
        model_refs=[maas_model_tinyllama_free.name],
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
) -> Generator[MaaSAuthPolicy]:
    applications_namespace = py_config["applications_namespace"]

    with MaaSAuthPolicy(
        client=admin_client,
        name="tinyllama-premium-access",
        namespace=applications_namespace,
        model_refs=[maas_model_tinyllama_premium.name],
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
) -> Generator[MaaSSubscription]:
    applications_namespace = py_config["applications_namespace"]

    with MaaSSubscription(
        client=admin_client,
        name="tinyllama-free-subscription",
        namespace=applications_namespace,
        owner={
            "groups": [{"name": maas_free_group}],
        },
        model_refs=[
            {
                "name": maas_model_tinyllama_free.name,
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
) -> Generator[MaaSSubscription]:
    applications_namespace = py_config["applications_namespace"]

    with MaaSSubscription(
        client=admin_client,
        name="tinyllama-premium-subscription",
        namespace=applications_namespace,
        owner={
            "groups": [{"name": maas_premium_group}],
        },
        model_refs=[
            {
                "name": maas_model_tinyllama_premium.name,
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
    LOGGER.info("MaaS: constructed model_url=%s (deployment=%s)", url, deployment_name)
    return url


@pytest.fixture(scope="class")
def model_url_tinyllama_premium(
    maas_scheme: str,
    maas_host: str,
    maas_inference_service_tinyllama_premium: LLMInferenceService,
) -> str:
    deployment_name = maas_inference_service_tinyllama_premium.name
    url = f"{maas_scheme}://{maas_host}/llm/{deployment_name}{CHAT_COMPLETIONS}"
    LOGGER.info("MaaS: constructed model_url=%s (deployment=%s)", url, deployment_name)
    return url


@pytest.fixture(scope="class")
def maas_api_key_for_actor(
    request_session_http: requests.Session,
    base_url: str,
    ocp_token_for_actor: str,
    maas_controller_enabled_latest: None,
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
