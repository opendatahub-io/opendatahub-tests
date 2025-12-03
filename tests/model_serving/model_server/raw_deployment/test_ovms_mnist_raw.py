import pytest

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import (
    KServeDeploymentType,
    ModelFormat,
    ModelVersion,
    Protocols,
    RuntimeTemplates,
)
from utilities.inference_utils import Inference
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG

pytestmark = [pytest.mark.rawdeployment, pytest.mark.usefixtures("valid_aws_config")]


@pytest.mark.smoke
@pytest.mark.jira("RHOAIENG-11749")
@pytest.mark.parametrize(
    "unprivileged_model_namespace, serving_runtime_from_template, s3_models_inference_service",
    [
        pytest.param(
            {"name": "raw-deployment-ovms-mnist"},
            {
                "name": f"{Protocols.HTTP}-{ModelFormat.ONNX}",
                "template-name": RuntimeTemplates.OVMS_KSERVE,
                "multi-model": False,
                "enable-http": True,
            },
            {
                "name": f"{Protocols.HTTP}-{ModelFormat.ONNX}",
                "deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT,
                "model-dir": "test-dir",
                "model-version": ModelVersion.OPSET13,
            },
        )
    ],
    indirect=True,
)
class TestOvmsMnistRaw:
    def test_ovms_mnist_inference_raw_internal_route(self, s3_models_inference_service):
        """Test OVMS MNIST model inference using internal route"""
        verify_inference_response(
            inference_service=s3_models_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTP,
            use_default_query=True,
        )
