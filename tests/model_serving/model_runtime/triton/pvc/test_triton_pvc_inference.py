"""
Test module for Triton PVC-based model deployments.

Validates Triton inference using PVC storage for models.
"""

import pytest
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod

from tests.model_serving.model_runtime.triton.basic_model_deployment.utils import (
    load_json,
    validate_inference_request,
)
from tests.model_serving.model_runtime.triton.constant import (
    BASE_RAW_DEPLOYMENT_CONFIG,
    MODEL_PATH_PREFIX,
    TRITON_REST_ONNX_INPUT_PATH,
)
from utilities.constants import Protocols

MODEL_PATH: str = f"{MODEL_PATH_PREFIX}"
ONNX_MODEL_NAME = "densenetonnx"

MODEL_STORAGE_URI_DICT = {"model-dir": MODEL_PATH}

pytestmark = pytest.mark.usefixtures(
    "root_dir",
    "valid_aws_config",
    "skip_if_no_supported_accelerator_type",
)


@pytest.mark.gpu
@pytest.mark.tier1
@pytest.mark.parametrize(
    (
        "protocol, model_namespace, triton_model_pvc, "
        "triton_pvc_downloaded_model_data, triton_pvc_serving_runtime, "
        "triton_pvc_inference_service"
    ),
    [
        pytest.param(
            {"protocol_type": Protocols.REST},
            {"name": "triton-pvc-onnx"},
            {"pvc-size": "10Gi"},
            MODEL_STORAGE_URI_DICT,
            {**BASE_RAW_DEPLOYMENT_CONFIG},
            {
                **BASE_RAW_DEPLOYMENT_CONFIG,
                "gpu_count": 0,
                "name": "triton-pvc-onnx-standard",
            },
            id="test_triton_pvc_onnx_standard",
        ),
    ],
    indirect=True,
)
class TestTritonPvcOnnxInference:
    """Validate Triton ONNX model inference from PVC-backed storage.

    Steps:
        1. Create a PVC and download the ONNX model from S3 into it.
        2. Deploy a Triton InferenceService using PVC storage.
        3. Run REST inference requests.
        4. Validate that inference responses contain expected content.
    """

    def test_triton_pvc_onnx_inference(
        self,
        triton_pvc_inference_service: InferenceService,
        triton_pod_resource: Pod,
        root_dir: str,
    ) -> None:
        """Given a Triton ISVC backed by PVC storage with the ONNX model,
        When REST inference requests are sent,
        Then the model returns valid responses.
        """
        input_query = load_json(path=TRITON_REST_ONNX_INPUT_PATH)

        validate_inference_request(
            pod_name=triton_pod_resource.name,
            isvc=triton_pvc_inference_service,
            input_query=input_query,
            model_name=ONNX_MODEL_NAME,
            protocol=Protocols.REST,
            root_dir=root_dir,
        )
