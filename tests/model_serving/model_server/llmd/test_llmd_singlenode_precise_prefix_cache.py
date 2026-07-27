import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.llm_inference_service import LLMInferenceService
from ocp_resources.prometheus import Prometheus

from tests.model_serving.model_server.llmd.llmd_configs import (
    PrecisePrefixCacheProducerConfig,
    PrecisePrefixCacheScorerConfig,
)
from tests.model_serving.model_server.llmd.utils import (
    assert_prefix_cache_routing,
    assert_scheduler_routing,
    get_llmd_inference_pool_pods,
    get_llmd_router_scheduler_pod,
    get_llmd_vllm_pods,
    ns_from_file,
    send_prefix_cache_requests,
)

NUM_REQUESTS = 12
PREFIX_CACHE_PROMPT = (
    "Explain in detail the fundamental principles of quantum mechanics including "
    "wave-particle duality, superposition, and entanglement in simple terms. "
    "Additionally, describe how these quantum phenomena differ from classical physics "
    "and why they are important for understanding the nature of reality at the atomic scale."
)

NAMESPACE = ns_from_file(file=__file__)

pytestmark = [pytest.mark.llmd_gpu]


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llmisvc",
    [
        pytest.param({"name": NAMESPACE}, PrecisePrefixCacheScorerConfig, id="scorer"),
        pytest.param({"name": NAMESPACE}, PrecisePrefixCacheProducerConfig, id="producer"),
    ],
    indirect=True,
)
@pytest.mark.usefixtures("valid_aws_config")
class TestSingleNodePrecisePrefixCache:
    """Deploy TinyLlama on GPU with 2 replicas and precise prefix cache routing,
    then verify cache hits via Prometheus metrics.

    Two EPP plugin variants are tested:
    - scorer: precise-prefix-cache-scorer (inference.networking.x-k8s.io/v1alpha1).
      Handles tokenization, KV indexing, and scoring in one plugin.
      Used until RHOAI 3.4, must remain supported for backward compatibility.
    - producer: precise-prefix-cache-producer (llm-d.ai/v1alpha1).
      Splits responsibilities across token-producer, precise-prefix-cache-producer,
      and prefix-cache-scorer plugins. Introduced in RHOAI 3.5.
    """

    def test_singlenode_precise_prefix_cache(
        self,
        request: pytest.FixtureRequest,
        unprivileged_client: DynamicClient,
        llmisvc: LLMInferenceService,
        llmisvc_token: str,
        prometheus: Prometheus,
    ) -> None:
        """Test steps:

        1. Assert the router-scheduler pod exists and is Running.
        2. Assert exactly 2 workload pods are found.
        3. Send identical chat completion requests with a shared long prompt.
        4. Query Prometheus and assert all traffic was routed to a single pod with correct prefix cache hit counts.
        5. Assert the scheduler made at least the expected number of routing decisions.
        """
        config = request.node.callspec.params["llmisvc"]

        router_pod = get_llmd_router_scheduler_pod(client=unprivileged_client, llmisvc=llmisvc)
        assert router_pod is not None, "Router-scheduler pod should exist"
        assert router_pod.instance.status.phase == "Running", "Router-scheduler pod should be running"

        vllm_pods = get_llmd_vllm_pods(client=unprivileged_client, llmisvc=llmisvc)
        inferencepool_pods = get_llmd_inference_pool_pods(client=unprivileged_client, llmisvc=llmisvc)
        # Single-node: all vLLM pods are InferencePool members (no headless workers).
        assert len(vllm_pods) == config.expected_vllm_pod_count, (
            f"Expected {config.expected_vllm_pod_count} vLLM pods, found {len(vllm_pods)}"
        )
        assert len(inferencepool_pods) == config.expected_inference_pool_pod_count, (
            f"Expected {config.expected_inference_pool_pod_count} InferencePool pods, found {len(inferencepool_pods)}"
        )
        assert len(vllm_pods) == len(inferencepool_pods), "Single-node: all vLLM pods should be InferencePool members"

        successful = send_prefix_cache_requests(
            llmisvc=llmisvc,
            prompt=PREFIX_CACHE_PROMPT,
            token=llmisvc_token,
            count=NUM_REQUESTS,
            delay_after_first_request=15,
        )

        assert_prefix_cache_routing(
            prometheus=prometheus,
            llmisvc=llmisvc,
            pods=inferencepool_pods,
            expected_requests=successful,
            block_size=request.node.callspec.params["llmisvc"].block_size,
        )
        assert_scheduler_routing(router_pod=router_pod, min_decisions=successful)
