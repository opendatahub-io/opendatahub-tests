import pytest

from tests.model_serving.model_server.llmd.utils import verify_llm_service_status, verify_gateway_status
from utilities.constants import Protocols
from utilities.llmd_utils import verify_inference_response_llmd

from utilities.llmd_constants import ModelStorage, ModelNames
from utilities.manifests.qwen2_7b_instruct_gpu import QWEN2_7B_INSTRUCT_GPU_INFERENCE_CONFIG

pytestmark = [
    pytest.mark.llmd_gpu,
    pytest.mark.gpu,
    pytest.mark.model_server_gpu,
]

GPU_LLMD_PARAMS = [
    pytest.param({"name": "llmd-gpu-standard"}, {"name_suffix": "gpu-standard"}, id="gpu-standard"),
    pytest.param(
        {"name": "llmd-gpu-no-scheduler"},
        {"name_suffix": "gpu-no-scheduler", "disable_scheduler": True},
        id="gpu-no-scheduler",
    ),
    pytest.param(
        {"name": "llmd-gpu-pd"},
        {
            "name_suffix": "gpu-pd",
            "enable_prefill_decode": True,
            "replicas": 2,
            "prefill_replicas": 1,
            "storage_uri": ModelStorage.HF_TINYLLAMA,
            "model_name": ModelNames.TINYLLAMA,
        },
        id="gpu-prefill-decode",
    ),
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

    """

    def test_llmd_hf_gpu_variants(self, llmd_gateway, llmd_inference_service_gpu, request, gpu_count_on_cluster):
        if gpu_count_on_cluster < 1:
            pytest.skip("No GPUs available on cluster, skipping GPU test")

        assert verify_gateway_status(llmd_gateway), "Gateway should be ready"
        assert verify_llm_service_status(llmd_inference_service_gpu), "LLMInferenceService should be ready"

        service = llmd_inference_service_gpu
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
