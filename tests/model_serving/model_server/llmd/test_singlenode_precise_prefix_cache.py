import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.llm_inference_service import LLMInferenceService

from tests.model_serving.model_server.llmd.utils import (
    get_llmd_router_scheduler_pod,
    get_llmd_workload_pods,
    verify_gateway_status,
    verify_llm_service_status,
    verify_singlenode_prefix_cache_routing,
)
from simple_logger.logger import get_logger

"""
Test Single-Node Precise Prefix Caching.

This test verifies that the LLM-D router correctly routes inference requests
based on cache state, maximizing prefix cache hits.

Test configuration:
- LLMInferenceService with 2 replicas and router enabled
- Authentication enabled
- Verify router pod and vLLM pods are running
- Send multiple requests with shared prefixes and size greater than PREFIX_CACHE_BLOCK_SIZE
"""

LOGGER = get_logger(name=__name__)

pytestmark = [pytest.mark.llmd_gpu]

@pytest.mark.parametrize(
    "unprivileged_model_namespace",
    [pytest.param({"name": "singlenode-prefix-cache-test"})],
    indirect=True,
)
@pytest.mark.usefixtures("valid_aws_config")
class TestSingleNodePrecisePrefixCache:
    """Test class for singlenode precise prefix cache routing."""

    @pytest.fixture(scope="class", autouse=True)
    def setup_auth(
        self,
        llmd_gateway,
        singlenode_precise_prefix_cache,
        llmd_s3_service_account,
        llmisvc_auth_token,
        llmisvc_auth_view_role,
        llmisvc_auth_role_binding,
    ):
        """Set up authentication for single-node prefix cache test."""
        # Create token with RBAC resources using factory fixtures
        token = llmisvc_auth_token(
            service_account=llmd_s3_service_account,
            llmisvc=singlenode_precise_prefix_cache,
            view_role_factory=llmisvc_auth_view_role,
            role_binding_factory=llmisvc_auth_role_binding,
        )

        # Store token as class attribute for use in tests
        TestSingleNodePrecisePrefixCache.auth_token = token

    def test_singlenode_precise_prefix_cache(
        self,
        unprivileged_client: DynamicClient,
        llmd_gateway,
        singlenode_precise_prefix_cache: LLMInferenceService,
        gpu_count_on_cluster: int,
    ):
        """Test single-node precise prefix cache routing."""
        if gpu_count_on_cluster < 2:
            pytest.skip(f"Test requires at least 2 GPUs (found {gpu_count_on_cluster})")

        # Verify gateway and service are ready
        assert verify_gateway_status(llmd_gateway), "Gateway should be ready"
        assert verify_llm_service_status(singlenode_precise_prefix_cache), "LLMInferenceService should be ready"

        # Verify router-scheduler pod exists and is running
        router_scheduler_pod = get_llmd_router_scheduler_pod(unprivileged_client, singlenode_precise_prefix_cache)
        assert router_scheduler_pod is not None, "Router-scheduler pod should exist"
        assert router_scheduler_pod.instance.status.phase == "Running", "Router-scheduler pod should be running"

        # Verify workload pods
        workload_pods = get_llmd_workload_pods(unprivileged_client, singlenode_precise_prefix_cache)
        assert len(workload_pods) == 2, f"Expected 2 workload pods, found {len(workload_pods)}"

        # Test prefix cache routing (includes assertions for routing affinity)
        verify_singlenode_prefix_cache_routing(
            llmisvc=singlenode_precise_prefix_cache,
            token=self.auth_token,
            workload_pods=workload_pods,
        )
