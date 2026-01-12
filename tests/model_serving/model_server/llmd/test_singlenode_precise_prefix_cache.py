"""
Test Single-Node Precise Prefix Caching.

This test verifies that the LLM-D router correctly routes inference requests
based on precise KV cache tracking, maximizing prefix cache hits using the
cache_tracking mode.

Test configuration:
- LLMInferenceService with 2 replicas and router enabled
- Authentication enabled
- Precise prefix cache mode with KV block index tracking
- HuggingFace model storage (TinyLlama)
- Verify router pod and vLLM pods are running
- Send multiple requests with shared prefixes and size greater than PREFIX_CACHE_BLOCK_SIZE
- Validate prefix cache metrics
"""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.gateway import Gateway
from ocp_resources.llm_inference_service import LLMInferenceService
from ocp_resources.prometheus import Prometheus

from tests.model_serving.model_server.llmd.utils import (
    get_llmd_router_scheduler_pod,
    get_llmd_workload_pods,
    send_prefix_cache_test_requests,
    verify_precise_prefix_cache,
    verify_gateway_status,
    verify_llm_service_status,
)

# Number of requests to send for prefix cache testing
NUM_REQUESTS = 20

pytestmark = [pytest.mark.llmd_gpu]


@pytest.mark.parametrize(
    "unprivileged_model_namespace, authenticated_llmisvc_token",
    [
        pytest.param(
            {"name": "llmd-test-singlenode-precise-prefix-cache"},
            {
                "service_account_fixture": "llmd_s3_service_account",
                "llmisvc_fixture": "singlenode_precise_prefix_cache",
            },
        )
    ],
    indirect=True,
)
@pytest.mark.usefixtures("valid_aws_config", "user_workload_monitoring_config_map")
class TestSingleNodePrecisePrefixCache:
    """Test class for singlenode precise prefix cache routing."""

    # @pytest.mark.skip(reason="Test currently failing - under investigation for precise prefix cache routing issues")
    def test_singlenode_precise_prefix_cache(
        self,
        unprivileged_client: DynamicClient,
        llmd_gateway: Gateway,
        singlenode_precise_prefix_cache: LLMInferenceService,
        authenticated_llmisvc_token: str,
        gpu_count_on_cluster: int,
        prometheus: Prometheus,
    ):
        """
        Test single-node precise prefix cache routing with KV block index tracking.

        This test validates:
        1. Scheduler deployment succeeds with precise mode configuration
        2. KV-cache routing correctly enabled for precise mode
        3. Requests with shared prefixes properly reuse cached responses
        4. KV block index metrics are exposed and show cache activity
        5. No redundant computation for repeated prefixes

        TODO: Fix precise prefix cache test - currently skipped due to bugs.
        """
        if gpu_count_on_cluster < 2:
            pytest.skip(f"Test requires at least 2 GPUs (found {gpu_count_on_cluster})")

        # Verify infrastructure is ready before testing routing
        assert verify_gateway_status(llmd_gateway), "Gateway should be ready"
        assert verify_llm_service_status(singlenode_precise_prefix_cache), "LLMInferenceService should be ready"

        router_scheduler_pod = get_llmd_router_scheduler_pod(
            client=unprivileged_client, llmisvc=singlenode_precise_prefix_cache
        )
        assert router_scheduler_pod is not None, "Router-scheduler pod should exist"
        assert router_scheduler_pod.instance.status.phase == "Running", "Router-scheduler pod should be running"

        workload_pods = get_llmd_workload_pods(client=unprivileged_client, llmisvc=singlenode_precise_prefix_cache)
        assert len(workload_pods) == 2, f"Expected 2 workload pods, found {len(workload_pods)}"

        import pdb; pdb.set_trace()
        # send_prefix_cache_test_requests(llmisvc=singlenode_precise_prefix_cache,token=authenticated_llmisvc_token,num_requests=3)

        # Send N identical requests to test prefix cache
        num_successful_requests = send_prefix_cache_test_requests(
            llmisvc=singlenode_precise_prefix_cache,
            token=authenticated_llmisvc_token,
            num_requests=NUM_REQUESTS,
        )

        # Verify precise prefix cache routing
        verify_precise_prefix_cache(
            prometheus=prometheus,
            llmisvc=singlenode_precise_prefix_cache,
            workload_pods=workload_pods,
            router_scheduler_pod=router_scheduler_pod,
            expected_requests=num_successful_requests,
        )
