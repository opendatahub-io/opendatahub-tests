import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.llm_inference_service import LLMInferenceService
from ocp_resources.prometheus import Prometheus

from tests.model_serving.model_server.llmd.llmd_configs import EstimatedPrefixCacheConfig
from tests.model_serving.model_server.llmd.utils import (
    assert_prefix_cache_routing,
    get_llmd_router_scheduler_pod,
    get_llmd_workload_pods,
    ns_from_file,
    send_prefix_cache_requests,
)

NUM_REQUESTS = 20
PREFIX_CACHE_PROMPT = (
    "Explain in detail the fundamental principles of quantum mechanics including "
    "wave-particle duality, superposition, and entanglement in simple terms. "
    "Additionally, describe how these quantum phenomena differ from classical physics "
    "and why they are important for understanding the nature of reality at the atomic scale."
)

NAMESPACE = ns_from_file(file=__file__)

pytestmark = [pytest.mark.tier2, pytest.mark.gpu]


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llmisvc",
    [({"name": NAMESPACE}, EstimatedPrefixCacheConfig)],
    indirect=True,
)
@pytest.mark.usefixtures("valid_aws_config", "skip_if_less_than_2_gpus", "skip_if_disconnected")
class TestSingleNodeEstimatedPrefixCache:
    """Deploy Qwen on GPU with 2 replicas and estimated prefix cache routing,
    then verify cache hits via Prometheus metrics.
    """

    def test_singlenode_estimated_prefix_cache(
        self,
        unprivileged_client: DynamicClient,
        llmisvc: LLMInferenceService,
        llmisvc_token: str,
        prometheus: Prometheus,
    ):
        """Test steps:

        1. Assert the router-scheduler pod exists and is Running.
        2. Assert exactly 2 workload pods are found.
        3. Send 20 chat completion requests with a shared long prompt.
        4. Query Prometheus and assert all traffic was routed to a single pod with correct prefix cache hit counts.
        """
        router_pod = get_llmd_router_scheduler_pod(client=unprivileged_client, llmisvc=llmisvc)
        assert router_pod is not None, "Router-scheduler pod should exist"
        assert router_pod.instance.status.phase == "Running", "Router-scheduler pod should be running"

        workload_pods = get_llmd_workload_pods(client=unprivileged_client, llmisvc=llmisvc)
        assert len(workload_pods) == 2, f"Expected 2 workload pods, found {len(workload_pods)}"

        successful = send_prefix_cache_requests(
            llmisvc=llmisvc,
            prompt=PREFIX_CACHE_PROMPT,
            token=llmisvc_token,
            count=NUM_REQUESTS,
        )
        assert successful == NUM_REQUESTS, f"Expected all {NUM_REQUESTS} requests to succeed, got {successful}"

        assert_prefix_cache_routing(
            prometheus=prometheus,
            llmisvc=llmisvc,
            pods=workload_pods,
            expected_requests=successful,
            block_size=EstimatedPrefixCacheConfig.block_size,
        )
