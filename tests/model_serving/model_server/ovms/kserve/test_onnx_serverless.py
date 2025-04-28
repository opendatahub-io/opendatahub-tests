import pytest

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import (
    KServeDeploymentType,
    ModelFormat,
    ModelVersion,
    Protocols,
    RunTimeConfigs,
)
from utilities.inference_utils import Inference
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG

pytestmark = [pytest.mark.serverless, pytest.mark.usefixtures("valid_aws_config")]


@pytest.mark.serverless
@pytest.mark.parametrize(
    "unprivileged_model_namespace, ovms_kserve_serving_runtime, ovms_kserve_inference_service",
    [
        pytest.param(
            {"name": "kserve-serverless-onnx"},
            RunTimeConfigs.ONNX_OPSET13_RUNTIME_CONFIG,
            {
                "name": ModelFormat.ONNX,
                "model-version": ModelVersion.OPSET13,
                "model-dir": "test-dir",
                "deployment-mode": KServeDeploymentType.SERVERLESS,
            },
        )
    ],
    indirect=True,
)
class TestONNXServerless:
    @pytest.mark.smoke
    @pytest.mark.jira("RHOAIENG-9045")
    def test_serverless_onnx_rest_inference(self, ovms_kserve_inference_service):
        """Verify that kserve Serverless ONNX model can be queried using REST"""
        verify_inference_response(
            inference_service=ovms_kserve_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )
