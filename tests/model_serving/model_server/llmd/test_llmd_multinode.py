import pytest

from tests.model_serving.model_server.llmd.utils import (
    # verify_gateway_status,
    verify_llm_service_status,
    verify_llmd_no_failed_pods,
)
from utilities.constants import Protocols
from utilities.llmd_utils import verify_inference_response_llmd
from utilities.manifests.tinyllama import TINYLLAMA_INFERENCE_CONFIG
from utilities.manifests.deepseek_coder_v2_lite import DEEPSEEK_CODER_V2_INFERENCE_CONFIG

pytestmark = [
    pytest.mark.llmd_gpu,
    pytest.mark.gpu,
    pytest.mark.model_server_gpu,
]


@pytest.mark.parametrize(
    "unprivileged_model_namespace",
    [pytest.param({"name": "llmd-multinode-rdma-test"})],
    indirect=True,
)
class TestMultiNodeLLMISVCRDMA:
    """Multi-Node LLMISVC test cases using RDMA networking."""

    def test_dp_ep(self, unprivileged_client, llmd_gateway, llmisvc_multinode_dp_ep):
        """Test multi-node llmisvc with DP + EP using RDMA (data: 32)."""
        llmisvc = llmisvc_multinode_dp_ep

        # assert verify_gateway_status(llmd_gateway), "Gateway should be ready"
        assert verify_llm_service_status(llmisvc), "LLMInferenceService should be ready"

        verify_inference_response_llmd(
            llm_service=llmisvc,
            inference_config=TINYLLAMA_INFERENCE_CONFIG,
            inference_type="completions",
            protocol=Protocols.HTTPS,
            use_default_query=True,
            insecure=True,
            model_name=llmisvc.name,
        )

        verify_llmd_no_failed_pods(client=unprivileged_client, llm_service=llmisvc)

    def test_dp_ep_prefill_decode(self, unprivileged_client, llmd_gateway, llmisvc_multinode_dp_ep_prefill_decode):
        """Test multi-node llmisvc with DP + EP and prefill-decode separation (RDMA)."""
        llmisvc = llmisvc_multinode_dp_ep_prefill_decode

        # assert verify_gateway_status(llmd_gateway), "Gateway should be ready"
        assert verify_llm_service_status(llmisvc), "LLMInferenceService should be ready"

        verify_inference_response_llmd(
            llm_service=llmisvc,
            inference_config=TINYLLAMA_INFERENCE_CONFIG,
            inference_type="completions",
            protocol=Protocols.HTTPS,
            use_default_query=True,
            insecure=True,
            model_name=llmisvc.name,
        )

        verify_llmd_no_failed_pods(client=unprivileged_client, llm_service=llmisvc)


@pytest.mark.parametrize(
    "unprivileged_model_namespace",
    [pytest.param({"name": "llmd-multinode-tcp-test"})],
    indirect=True,
)
class TestMultiNodeLLMISVCTCP:
    """Multi-Node LLMISVC test cases using TCP networking (no RDMA)."""

    def test_dp_ep(self, unprivileged_client, llmd_gateway, llmisvc_multinode_dp_ep_tcp):
        """Test multi-node llmisvc with DP + EP using TCP (data: 16)."""
        llmisvc = llmisvc_multinode_dp_ep_tcp

        # assert verify_gateway_status(llmd_gateway), "Gateway should be ready"
        assert verify_llm_service_status(llmisvc), "LLMInferenceService should be ready"

        verify_inference_response_llmd(
            llm_service=llmisvc,
            inference_config=DEEPSEEK_CODER_V2_INFERENCE_CONFIG,
            inference_type="completions",
            protocol=Protocols.HTTPS,
            use_default_query=True,
            insecure=True,
            model_name=llmisvc.name,
        )

        verify_llmd_no_failed_pods(client=unprivileged_client, llm_service=llmisvc)
