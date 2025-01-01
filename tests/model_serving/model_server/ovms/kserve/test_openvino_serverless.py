import pytest

from tests.model_serving.model_server.authentication.utils import (
    verify_inference_response,
)
from utilities.constants import (
    ModelAndFormat,
    ModelFormat,
    ModelStoragePath,
    ModelVersion,
    Protocols,
    ModelInferenceRuntime,
)
from utilities.inference_utils import Inference


@pytest.mark.parametrize(
    "model_namespace, ci_s3_storage_uri, openvino_kserve_serving_runtime, ovms_serverless_inference_service",
    [
        pytest.param(
            {"name": "kserve-serverless-openvino"},
            {"model-dir": ModelStoragePath.KSERVE_OPENVINO_EXAMPLE_MODEL},
            {
                "runtime-name": ModelInferenceRuntime.OPENVINO_KSERVE_RUNTIME,
                "model-format": {ModelAndFormat.OPENVINO_IR: ModelVersion.OPSET1},
            },
            {"name": ModelFormat.OPENVINO, "model-version": ModelVersion.OPSET1},
        )
    ],
    indirect=True,
)
class TestOpenVINO:
    @pytest.mark.smoke
    @pytest.mark.polarion("ODS-2626")
    def test_serverless_openvino_rest_inference(self, ovms_serverless_inference_service):
        """Verify that kserve Serverless OpenVINO model can be queried using REST"""
        verify_inference_response(
            inference_service=ovms_serverless_inference_service,
            runtime=ModelInferenceRuntime.OPENVINO_KSERVE_RUNTIME,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )
