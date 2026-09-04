from typing import Any

import pytest
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_runtime.vllm.constant import (
    COMPLETION_QUERY,
    GRANITE_CHAT_QUERY,
    GRANITE_VISION_SPYRE_SERVING_ARGUMENT,
    PREDICT_RESOURCES_VISION,
)
from tests.model_serving.model_runtime.vllm.utils import validate_raw_openai_inference_request
from utilities.constants import KServeDeploymentType

SERVING_ARGUMENT: list[str] = [
    "--model=/mnt/models",
    "--uvicorn-log-level=debug",
    "--dtype=float16",
]

MODEL_PATH: str = "models/granite-vision-3.3-2b"

PVC_RAW_DEPLOYMENT_CONFIG: dict[str, Any] = {
    "deployment_mode": KServeDeploymentType.STANDARD,
    "runtime_argument": SERVING_ARGUMENT,
    "min-replicas": 1,
}

pytestmark = pytest.mark.usefixtures("skip_if_no_supported_accelerator_type", "valid_aws_config")


@pytest.mark.vllm_spyreppc64le_gpu
@pytest.mark.vllm_nvidia_multi_gpu
@pytest.mark.vllm_amd_gpu
@pytest.mark.parametrize(
    "model_namespace, vllm_model_pvc, pvc_downloaded_model_data, serving_runtime, vllm_pvc_inference_service",
    [
        pytest.param(
            {"name": "vllm-pvc-granite-vision-2b"},
            {"pvc-size": "100Gi"},
            {"model-dir": MODEL_PATH},
            {"deployment_mode": KServeDeploymentType.STANDARD},
            {
                "deployment_mode": KServeDeploymentType.STANDARD,
                "runtime_argument": GRANITE_VISION_SPYRE_SERVING_ARGUMENT,
                "min-replicas": 1,
                "gpu_count": 2,
                "predict_resources": PREDICT_RESOURCES_VISION,
                "name": "vllm-pvc-granite-vision-2b",
                "timeout": 1200,
            },
            id="test_vllm_pvc_granite_vision_2b_spyre_ppc64le",
        ),
    ],
    indirect=True,
)
class TestVllmPvcGraniteVision2BInference:
    """Validate vLLM Granite-Vision-3.3-2b model inference from PVC-backed storage.

    Steps:
        1. Create a PVC and download the Granite-Vision-3.3-2b model from S3 into it.
        2. Deploy a vLLM InferenceService using PVC storage.
        3. Run OpenAI-compatible chat and completion requests.
        4. Validate that inference responses contain expected content.
    """

    def test_vllm_pvc_granite_vision_2b_openai_inference(
        self,
        vllm_pvc_inference_service: InferenceService,
        response_snapshot: Any,
    ) -> None:
        """Given a vLLM ISVC backed by PVC storage with Granite-Vision-3.3-2b,
        When OpenAI-compatible chat and completion requests are sent over the external route,
        Then the model returns valid responses.
        """
        validate_raw_openai_inference_request(
            isvc=vllm_pvc_inference_service,
            response_snapshot=response_snapshot,
            chat_query=GRANITE_CHAT_QUERY,
            completion_query=COMPLETION_QUERY,
        )
