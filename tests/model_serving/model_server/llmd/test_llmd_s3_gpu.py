import pytest

from tests.model_serving.model_server.llmd.utils import (
    verify_gateway_status,
    verify_llm_service_status,
)
from utilities.constants import Protocols
from utilities.llmd_constants import ModelNames, ModelStorage
from utilities.llmd_utils import verify_inference_response_llmd
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
            "storage_uri": ModelStorage.S3_QWEN,
            "model_name": ModelNames.QWEN,
        },
        id="gpu-prefill-decode",
    ),
]


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llmd_inference_service_gpu",
    GPU_LLMD_PARAMS,
    indirect=True,
)
@pytest.mark.usefixtures("valid_aws_config")
class TestLLMDS3GPUInference:
    """LLMD inference testing with S3 storage and GPU runtime using vLLM."""

    def test_llmd_s3_gpu(
        self, unprivileged_client, llmd_gateway, llmd_inference_service_gpu, request, gpu_count_on_cluster
    ):
        """Test LLMD inference with various GPU configurations using S3 storage."""
        if gpu_count_on_cluster < 1:
            pytest.skip("No GPUs available on cluster, skipping GPU test")

        assert verify_gateway_status(llmd_gateway), "Gateway should be ready"
        assert verify_llm_service_status(llmd_inference_service_gpu), "LLMInferenceService should be ready"

        verify_inference_response_llmd(
            llm_service=llmd_inference_service_gpu,
            inference_config=QWEN2_7B_INSTRUCT_GPU_INFERENCE_CONFIG,
            inference_type="chat_completions",
            protocol=Protocols.HTTP,
            use_default_query=True,
            insecure=True,
            model_name=llmd_inference_service_gpu.instance.spec.model.name,
        )
