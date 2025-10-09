import pytest

from tests.model_serving.model_server.llmd.utils import verify_llm_service_status, verify_gateway_status
from utilities.constants import Protocols
from utilities.llmd_utils import verify_inference_response_llmd

from utilities.llmd_constants import GPU_LLMD_PARAMS
from utilities.manifests.qwen2_7b_instruct_gpu import QWEN2_7B_INSTRUCT_GPU_INFERENCE_CONFIG

pytestmark = [
    pytest.mark.llmd_gpu,
    pytest.mark.gpu,
    pytest.mark.model_server_gpu,
]


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llmd_inference_service_gpu",
    GPU_LLMD_PARAMS,
    indirect=True,
)
class TestLLMDHFGPUInference:
    """LLMD inference testing with HuggingFace storage and GPU runtime using vLLM.
    
    Tests multiple GPU configurations based on temp YAML patterns using cluster LLMInferenceServiceConfigs:
    1. Standard GPU with scheduler (uses: kserve-config-llm-template + kserve-config-llm-scheduler)
    2. GPU without scheduler (uses: kserve-config-llm-template only)  
    3. GPU with prefill/decode (uses: kserve-config-llm-template + kserve-config-llm-scheduler + 
                                      kserve-config-llm-prefill-template + kserve-config-llm-decode-template)
    
    This approach:
    - Leverages existing cluster configurations instead of hardcoding
    - Reduces code duplication and maintenance overhead
    - Uses battle-tested platform configurations
    - Validates config reference functionality
    """

    def test_llmd_hf_gpu_variants(self, llmd_gateway, llmd_inference_service_gpu, request, gpu_count_on_cluster):
        """Test GPU inference with different configurations (E2E validation)."""
        # Skip test if no GPUs available on cluster
        if gpu_count_on_cluster < 1:
            pytest.skip("No GPUs available on cluster, skipping GPU test")
            
        # Verify core components are ready
        assert verify_gateway_status(llmd_gateway), "Gateway should be ready"
        assert verify_llm_service_status(llmd_inference_service_gpu), "LLMInferenceService should be ready"

        # Extract test variant information for inference
        test_params = request.node.callspec.params.get('llmd_inference_service_gpu', {})
        service = llmd_inference_service_gpu
        
        # Test inference - the core E2E validation
        inference_config = QWEN2_7B_INSTRUCT_GPU_INFERENCE_CONFIG
        verify_inference_response_llmd(
            llm_service=llmd_inference_service_gpu,
            inference_config=inference_config,
            inference_type="chat_completions",
            protocol=Protocols.HTTP,
            use_default_query=True,
            insecure=True,
            model_name=service.instance.spec.model.name,
        )


