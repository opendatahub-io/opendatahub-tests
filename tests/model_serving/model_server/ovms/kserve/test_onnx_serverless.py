import pytest

from tests.model_serving.model_server.authentication.utils import verify_inference_response
from utilities.constants import ModelFormat, ModelVersion, Protocols, ModelInferenceRuntime
from utilities.inference_utils import Inference


@pytest.mark.parametrize(
    "model_namespace, ci_s3_storage_uri, openvino_kserve_serving_runtime, http_openvino_serverless_inference_service",
    [
        pytest.param(
            {"name": "kserve-serverless-openvino"},
            {"model-dir": "test-dir"},
            {"model-format": {ModelFormat.ONNX: ModelVersion.OPSET13}},
            {"model-version": ModelVersion.OPSET13},
        )
    ],
    indirect=True,
)
class TestOpenVINO:
    @pytest.mark.smoke
    @pytest.mark.jira("RHOAIENG-9045")
    def test_serverless_onnx_rest_inference(self, http_openvino_serverless_inference_service):
        """Verify that kserve Serverless ONNX model can be queried using REST"""
        verify_inference_response(
            inference_service=http_openvino_serverless_inference_service,
            runtime=ModelInferenceRuntime.ONNX_RUNTIME,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            model_name=ModelFormat.ONNX,
            use_default_query=True,
        )
