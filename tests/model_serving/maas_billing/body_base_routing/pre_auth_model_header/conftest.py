"""Fixtures for body-based routing (BBR) integration tests."""

from collections.abc import Generator
from typing import Any

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.llm_inference_service import LLMInferenceService
from ocp_resources.maas_model_ref import MaaSModelRef
from ocp_resources.maas_subscription import MaaSSubscription
from ocp_resources.namespace import Namespace

from tests.model_serving.maas_billing.body_base_routing.pre_auth_model_header.utils import (
    BBR_PRE_PROCESSING_DEPLOYMENT_NAME,
    BBR_RATE_LIMIT_CHAT_MAX_TOKENS,
    BBR_RATE_LIMIT_TOKENS_PER_MINUTE,
    bbr_api_key_lifecycle,
    get_bbr_envoy_filter_config_patches,
)
from tests.model_serving.maas_billing.utils import build_maas_headers
from utilities.constants import MAAS_GATEWAY_NAMESPACE
from utilities.general import generate_random_name

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="session")
def bbr_gateway_namespace(admin_client: DynamicClient) -> str:
    """Return the gateway namespace where BBR infrastructure is deployed."""
    gateway_namespace = Namespace(client=admin_client, name=MAAS_GATEWAY_NAMESPACE)
    assert gateway_namespace.exists, (
        f"Gateway namespace '{MAAS_GATEWAY_NAMESPACE}' not found — "
        f"required for BBR tests ({BBR_PRE_PROCESSING_DEPLOYMENT_NAME} is deployed here)"
    )
    return MAAS_GATEWAY_NAMESPACE


@pytest.fixture(scope="class")
def bbr_envoy_filter_config_patches(
    admin_client: DynamicClient,
    bbr_gateway_namespace: str,
) -> list[Any]:
    """Return BBR EnvoyFilter configPatches once per test class to avoid redundant API calls."""
    return get_bbr_envoy_filter_config_patches(
        admin_client=admin_client,
        gateway_namespace=bbr_gateway_namespace,
    )


@pytest.fixture(scope="class")
def bbr_inference_url(
    maas_scheme: str,
    maas_host: str,
    maas_inference_service_tinyllama_free: LLMInferenceService,
) -> str:
    """BBR inference URL — model name present in both URL path and request body."""
    return f"{maas_scheme}://{maas_host}/llm/{maas_inference_service_tinyllama_free.name}/v1/chat/completions"


@pytest.fixture(scope="class")
def bbr_api_key_headers(bbr_valid_api_key: str) -> dict[str, str]:
    """Authorization headers for BBR inference requests using a valid API key."""
    return build_maas_headers(token=bbr_valid_api_key)


@pytest.fixture(scope="class")
def bbr_chat_payload(maas_inference_service_tinyllama_free: LLMInferenceService) -> dict[str, Any]:
    """Minimal chat completions payload for BBR inference tests with model name in body."""
    return {
        "model": maas_inference_service_tinyllama_free.name,
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 1,
    }


@pytest.fixture(scope="class")
def bbr_rate_limit_chat_payload(maas_inference_service_tinyllama_free: LLMInferenceService) -> dict[str, Any]:
    """Chat completions payload for BBR rate limit tests using a large max_tokens to exhaust the token quota quickly."""
    return {
        "model": maas_inference_service_tinyllama_free.name,
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": BBR_RATE_LIMIT_CHAT_MAX_TOKENS,
    }


@pytest.fixture(scope="class")
def bbr_valid_api_key(
    request_session_http: requests.Session,
    base_url: str,
    ocp_token_for_actor: str,
    maas_subscription_tinyllama_free: MaaSSubscription,
) -> Generator[str, Any, Any]:
    """Create an API key bound to the free TinyLlama subscription and yield the plaintext key."""
    with bbr_api_key_lifecycle(
        request_session_http=request_session_http,
        base_url=base_url,
        ocp_token_for_actor=ocp_token_for_actor,
        subscription_name=maas_subscription_tinyllama_free.name,
        key_name_prefix="e2e-bbr-inference",
        fixture_label="bbr_valid_api_key",
    ) as api_key:
        yield api_key


@pytest.fixture
def bbr_low_limit_subscription(
    admin_client: DynamicClient,
    maas_subscription_namespace: Namespace,
    maas_model_tinyllama_free: MaaSModelRef,
    maas_free_group: str,
) -> Generator[MaaSSubscription, Any, Any]:
    """Create a MaaSSubscription with a low token rate limit for BBR rate limiting tests."""
    sub_name = f"bbr-rate-limit-sub-{generate_random_name()}"
    with MaaSSubscription(
        client=admin_client,
        name=sub_name,
        namespace=maas_subscription_namespace.name,
        owner={"groups": [{"name": maas_free_group}]},
        model_refs=[
            {
                "name": maas_model_tinyllama_free.name,
                "namespace": maas_model_tinyllama_free.namespace,
                "tokenRateLimits": [{"limit": BBR_RATE_LIMIT_TOKENS_PER_MINUTE, "window": "1m"}],
            }
        ],
        priority=1,
        wait_for_resource=True,
    ) as subscription:
        subscription.wait_for_condition(condition="Ready", status="True", timeout=300)
        LOGGER.info(
            f"bbr_low_limit_subscription: created '{sub_name}' with token limit {BBR_RATE_LIMIT_TOKENS_PER_MINUTE}/min"
        )
        yield subscription


@pytest.fixture
def bbr_rate_limited_api_key(
    request_session_http: requests.Session,
    base_url: str,
    ocp_token_for_actor: str,
    bbr_low_limit_subscription: MaaSSubscription,
) -> Generator[str, Any, Any]:
    """API key bound to the low-rate-limit BBR subscription."""
    with bbr_api_key_lifecycle(
        request_session_http=request_session_http,
        base_url=base_url,
        ocp_token_for_actor=ocp_token_for_actor,
        subscription_name=bbr_low_limit_subscription.name,
        key_name_prefix="e2e-bbr-rate-limit",
        fixture_label="bbr_rate_limited_api_key",
    ) as api_key:
        yield api_key
