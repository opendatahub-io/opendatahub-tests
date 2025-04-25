import pytest

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import (
    ModelFormat,
    ModelVersion,
    Protocols,
)
from utilities.inference_utils import Inference
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG

pytestmark = [
    pytest.mark.serverless,
    pytest.mark.sanity,
    pytest.mark.usefixtures("valid_aws_config"),
]


NAME: str = "serverless-same-name"


@pytest.mark.serverless
@pytest.mark.parametrize(
    "model_namespace, ci_endpoint_s3_secret, ovms_kserve_serving_runtime, ovms_kserve_serverless_inference_service",
    [
        pytest.param(
            {"name": NAME},
            {"name": NAME},
            {
                "runtime-name": NAME,
                "model-format": {ModelFormat.ONNX: ModelVersion.OPSET13},
            },
            {
                "name": NAME,
                "model-version": ModelVersion.OPSET13,
                "model-dir": "test-dir",
                "enable-external-route": False,
            },
        )
    ],
    indirect=True,
)
class TestServerlessResourcesWithSameName:
    def test_serverless_resources_with_same_name(self, ovms_kserve_serverless_inference_service):
        """Verify model can be queried"""
        verify_inference_response(
            inference_service=ovms_kserve_serverless_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTP,
            use_default_query=True,
        )
