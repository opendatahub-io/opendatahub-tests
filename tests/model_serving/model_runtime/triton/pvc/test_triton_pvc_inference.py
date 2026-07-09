"""
Test module for Triton ONNX model using PVC storage with REST protocol.

Validates inference for Triton ONNX model using PVC-backed storage:
    - ONNX (densenetonnx)

Steps:
    1. Create a PVC and download the Triton model from S3 into it.
    2. Deploy a Triton InferenceService using PVC storage.
    3. Run KServe v2 REST inference requests.
    4. Validate that inference responses contain expected content.
"""

import pytest
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod

from tests.model_serving.model_runtime.triton.constant import (
    BASE_RAW_DEPLOYMENT_CONFIG,
    MODEL_PATH_PREFIX,
    TRITON_REST_ONNX_INPUT_PATH,
)
from tests.model_serving.model_runtime.triton.S3.utils import load_json, validate_inference_request
from utilities.constants import Protocols

pytestmark = pytest.mark.usefixtures("root_dir", "valid_aws_config")


@pytest.mark.parametrize(
    (
        "protocol",
        "model_namespace",
        "triton_model_pvc",
        "triton_pvc_downloaded_model_data",
        "triton_pvc_serving_runtime",
        "triton_pvc_inference_service",
        "model_name",
        "input_path",
    ),
    [
        # ONNX Model - PVC Storage
        pytest.param(
            {"protocol_type": Protocols.REST},
            {"name": "triton-pvc-onnx"},
            {"pvc-size": "10Gi"},
            {"model-dir": MODEL_PATH_PREFIX},
            {**BASE_RAW_DEPLOYMENT_CONFIG},
            {"name": "triton-pvc-onnx-rest", "gpu_count": 0, **BASE_RAW_DEPLOYMENT_CONFIG},
            "densenetonnx",
            TRITON_REST_ONNX_INPUT_PATH,
            id="onnx-pvc-rest",
            marks=pytest.mark.smoke,
        ),
    ],
    indirect=True,
)
class TestTritonPvcInference:
    """Validate Triton ONNX model inference from PVC-backed storage.

    This test class validates that Triton InferenceServices can successfully:
    1. Load ONNX models from PVC storage (instead of S3)
    2. Perform inference using the KServe v2 REST protocol
    3. Return valid responses

    The PVC storage pattern enables:
    - Faster model loading (no S3 network latency)
    - Offline/air-gapped deployments
    - Support for large models that may have S3 access limitations
    """

    def test_triton_pvc_inference(
        self,
        triton_pvc_inference_service: InferenceService,
        triton_pod_resource: Pod,
        model_name: str,
        input_path: str,
        protocol: str,
        root_dir: str,
    ) -> None:
        """Given a Triton ISVC backed by PVC storage with a model,
        When a KServe v2 REST inference request is sent,
        Then the model returns a valid response.
        """
        input_query = load_json(path=input_path)

        validate_inference_request(
            pod_name=triton_pod_resource.name,
            isvc=triton_pvc_inference_service,
            response_snapshot=None,
            input_query=input_query,
            model_name=model_name,
            protocol=protocol,
            root_dir=root_dir,
        )
