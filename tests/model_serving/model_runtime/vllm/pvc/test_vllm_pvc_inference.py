from typing import Any

import pytest
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_runtime.vllm.constant import (
    COMPLETION_QUERY,
    GRANITE_CHAT_QUERY,
    GRANITE_SERVING_ARGUMENT,
)
from tests.model_serving.model_runtime.vllm.utils import validate_raw_openai_inference_request
from utilities.constants import KServeDeploymentType

MODEL_PATH: str = "granite-7b-starter"

PVC_RAW_DEPLOYMENT_CONFIG: dict[str, Any] = {
    "deployment_mode": KServeDeploymentType.STANDARD,
    "runtime_argument": GRANITE_SERVING_ARGUMENT,
    "min-replicas": 1,
}

pytestmark = pytest.mark.usefixtures("skip_if_no_supported_accelerator_type", "valid_aws_config")


@pytest.mark.vllm_nvidia_single_gpu
@pytest.mark.vllm_amd_gpu
@pytest.mark.parametrize(
    "model_namespace, vllm_model_pvc, pvc_downloaded_model_data, serving_runtime, vllm_pvc_inference_service",
    [
        pytest.param(
            {"name": "vllm-pvc-granite"},
            {"pvc-size": "20Gi"},
            {"model-dir": MODEL_PATH},
            {"deployment_mode": KServeDeploymentType.STANDARD},
            {
                **PVC_RAW_DEPLOYMENT_CONFIG,
                "gpu_count": 1,
                "name": "vllm-pvc-granite",
            },
            id="test_vllm_pvc_granite_standard_single_gpu",
        ),
    ],
    indirect=True,
)
class TestVllmPvcGraniteInference:
    """Validate vLLM Granite model inference from PVC-backed storage.

    Steps:
        1. Create a PVC and download the Granite model from S3 into it.
        2. Deploy a vLLM InferenceService using PVC storage.
        3. Run OpenAI-compatible chat and completion requests.
        4. Validate that inference responses contain expected content.
    """

    def test_vllm_pvc_granite_openai_inference(
        self,
        vllm_pvc_inference_service: InferenceService,
        response_snapshot: Any,
    ) -> None:
        """Given a vLLM ISVC backed by PVC storage with the Granite model,
        When OpenAI-compatible chat and completion requests are sent over the external route,
        Then the model returns valid responses.
        """
        validate_raw_openai_inference_request(
            isvc=vllm_pvc_inference_service,
            response_snapshot=response_snapshot,
            chat_query=GRANITE_CHAT_QUERY,
            completion_query=COMPLETION_QUERY,
        )
