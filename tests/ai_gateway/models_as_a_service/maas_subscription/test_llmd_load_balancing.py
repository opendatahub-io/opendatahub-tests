from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.maas_subscription import MaaSSubscription
from ocp_resources.prometheus import Prometheus
from timeout_sampler import TimeoutSampler

from tests.ai_gateway.models_as_a_service.maas_subscription.utils import maas_body_routed_model_name
from tests.ai_gateway.models_as_a_service.utils import build_maas_headers, create_api_key, revoke_api_key

# The MaaS test exercises the generic LLM-d OpenAI SDK validator through an API-key-authenticated Gateway route.
from tests.model_serving.model_server.llmd.api_compat import (  # noqa: NIT001
    BearerTokenProvider,
    OpenAICompatibilityValidator,
)
from tests.model_serving.model_server.llmd.utils import (  # noqa: NIT001
    get_llmd_inference_pool_pods,
    get_llmd_router_scheduler_pod,
    query_metric_by_pod,
)
from utilities.general import generate_random_name
from utilities.resources.llm_inference_service import LLMInferenceService

LOGGER = structlog.get_logger(name=__name__)

CONCURRENT_REQUESTS = 8
REQUEST_TIMEOUT_SECONDS = 180
METRICS_TIMEOUT_SECONDS = 240


def _send_chat_completion(
    inference_url: str,
    headers: dict[str, str],
    model_name: str,
    request_number: int,
) -> tuple[int, str]:
    """Send one distinct, long-running chat completion request through MaaS."""
    response = requests.post(
        url=inference_url,
        headers=headers,
        json={
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Explain how a load balancer distributes requests across independent workers. "
                        f"Include this unique request number: {request_number}."
                    ),
                }
            ],
            "max_tokens": 128,
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
        verify=False,
    )
    return response.status_code, response.text[:200]


@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "maas_inference_service_tinyllama_load_balanced",
    "maas_model_tinyllama_load_balanced",
    "maas_auth_policy_tinyllama_load_balanced",
    "maas_subscription_tinyllama_load_balanced",
    "maas_api_gateway_reachable",
)
@pytest.mark.tier2
@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.llmd_gpu
class TestMaaSGatewayLlmdLoadBalancing:
    """Verify MaaS-authorized requests are distributed by LLM-d after gateway routing."""

    def test_unauthenticated_gateway_request_is_rejected(
        self,
        request_session_http: requests.Session,
        model_url_tinyllama_load_balanced: str,
        maas_inference_service_tinyllama_load_balanced: LLMInferenceService,
    ) -> None:
        """Given an auth-enabled MaaS LLM-d model, when a Gateway request has no API key,
        then MaaS rejects it before LLM-d can serve inference.
        """
        response = request_session_http.post(
            url=model_url_tinyllama_load_balanced,
            json={
                "model": maas_body_routed_model_name(llmisvc=maas_inference_service_tinyllama_load_balanced),
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 8,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
            verify=False,
        )
        assert response.status_code == 401, (
            "Expected MaaS Gateway to reject a request without an API key, "
            f"got {response.status_code}: {response.text[:200]}"
        )

    def test_llmd_scheduler_has_two_ready_pool_backends(
        self,
        admin_client: DynamicClient,
        maas_inference_service_tinyllama_load_balanced: LLMInferenceService,
    ) -> None:
        """Given a two-replica MaaS LLM-d service, when it is Ready,
        then its scheduler exposes two ready pool backends.
        """
        router_scheduler_pod = get_llmd_router_scheduler_pod(
            client=admin_client,
            llmisvc=maas_inference_service_tinyllama_load_balanced,
        )
        assert router_scheduler_pod is not None, "Expected an LLM-d router-scheduler pod"
        assert router_scheduler_pod.instance.status.phase == "Running", (
            f"Expected router-scheduler pod '{router_scheduler_pod.name}' to be Running"
        )

        pool_pods = get_llmd_inference_pool_pods(
            client=admin_client,
            llmisvc=maas_inference_service_tinyllama_load_balanced,
        )
        assert len(pool_pods) == 2, f"Expected two LLM-d pool backends, found {[pod.name for pod in pool_pods]}"
        assert all(pod.instance.status.phase == "Running" for pod in pool_pods), (
            f"Expected all pool backends to be Running, got "
            f"{[(pod.name, pod.instance.status.phase) for pod in pool_pods]}"
        )

    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_authorized_requests_are_distributed_across_llmd_backends(
        self,
        request_session_http: requests.Session,
        base_url: str,
        ocp_token_for_actor: str,
        admin_client: DynamicClient,
        prometheus: Prometheus,
        model_url_tinyllama_load_balanced: str,
        maas_inference_service_tinyllama_load_balanced: LLMInferenceService,
        maas_subscription_tinyllama_load_balanced: MaaSSubscription,
    ) -> None:
        """Given a subscription-authorized API key and two LLM-d backends,
        when concurrent requests cross the MaaS Gateway,
        then every backend receives successful inference traffic.
        """
        pool_pods = get_llmd_inference_pool_pods(
            client=admin_client,
            llmisvc=maas_inference_service_tinyllama_load_balanced,
        )
        assert len(pool_pods) == 2, f"Expected two LLM-d pool backends, found {[pod.name for pod in pool_pods]}"
        baseline_counts = query_metric_by_pod(
            prometheus=prometheus,
            metric_name="kserve_vllm:request_success_total",
            llmisvc=maas_inference_service_tinyllama_load_balanced,
            pods=pool_pods,
        )

        _, api_key = create_api_key(
            base_url=base_url,
            ocp_user_token=ocp_token_for_actor,
            request_session_http=request_session_http,
            api_key_name=f"e2e-llmd-load-balanced-{generate_random_name()}",
            subscription=maas_subscription_tinyllama_load_balanced.name,
        )
        try:
            headers = build_maas_headers(token=api_key["key"])
            model_name = maas_body_routed_model_name(llmisvc=maas_inference_service_tinyllama_load_balanced)
            with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor:
                responses = list(
                    executor.map(
                        lambda request_number: _send_chat_completion(
                            inference_url=model_url_tinyllama_load_balanced,
                            headers=headers,
                            model_name=model_name,
                            request_number=request_number,
                        ),
                        range(CONCURRENT_REQUESTS),
                    )
                )

            failed_responses = [response for response in responses if response[0] != 200]
            assert not failed_responses, (
                f"Expected all MaaS Gateway inference requests to succeed, got {failed_responses}"
            )

            latest_counts = baseline_counts
            for latest_counts in TimeoutSampler(
                wait_timeout=METRICS_TIMEOUT_SECONDS,
                sleep=10,
                func=query_metric_by_pod,
                prometheus=prometheus,
                metric_name="kserve_vllm:request_success_total",
                llmisvc=maas_inference_service_tinyllama_load_balanced,
                pods=pool_pods,
            ):
                request_deltas = {
                    pod_name: latest_counts[pod_name] - baseline_counts[pod_name] for pod_name in latest_counts
                }
                LOGGER.info(f"MaaS Gateway LLM-d request distribution: {request_deltas}")
                if all(request_deltas.get(pod.name, 0) > 0 for pod in pool_pods):
                    return

            pytest.fail(
                "Concurrent MaaS Gateway requests were not distributed to every LLM-d pool backend. "
                f"Baseline counts={baseline_counts}, final counts={latest_counts}"
            )
        finally:
            revoke_api_key(
                request_session_http=request_session_http,
                base_url=base_url,
                key_id=api_key["id"],
                ocp_user_token=ocp_token_for_actor,
            )

    # Run the broad OpenAI contract after the dedicated MaaS routing assertion.
    @pytest.mark.soak
    @pytest.mark.order(after="test_authorized_requests_are_distributed_across_llmd_backends")
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_authorized_openai_compatibility(
        self,
        ocp_token_for_actor: str,
        model_url_tinyllama_load_balanced: str,
        maas_inference_service_tinyllama_load_balanced: LLMInferenceService,
        api_key_bound_to_load_balanced_subscription: str,
    ) -> None:
        """Given a subscription-authorized MaaS API key, when LLM-d's OpenAI validator targets the Gateway route,
        then every OpenAI compatibility assertion, including tool calling, passes.
        """
        _ = ocp_token_for_actor
        gateway_base_url = model_url_tinyllama_load_balanced.removesuffix("/v1/chat/completions")
        model_name = maas_body_routed_model_name(llmisvc=maas_inference_service_tinyllama_load_balanced)
        api_key_provider = BearerTokenProvider(token=api_key_bound_to_load_balanced_subscription)

        with OpenAICompatibilityValidator(
            base_url=gateway_base_url,
            model_name=model_name,
            api_key_provider=api_key_provider,
            verify_ssl=False,
        ) as validator:
            validator.run_all()
