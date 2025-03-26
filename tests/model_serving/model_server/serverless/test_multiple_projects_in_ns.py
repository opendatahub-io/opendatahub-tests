import pytest

from tests.model_serving.model_server.serverless.constants import ONNX_SERVERLESS_INFERENCE_SERVICE_CONFIG
from tests.model_serving.model_server.utils import run_inference_multiple_times
from utilities.constants import (
    Protocols,
    RunTimeConfigs,
)
from utilities.inference_utils import Inference
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG

pytestmark = [pytest.mark.serverless, pytest.mark.sanity]


@pytest.mark.polarion("ODS-2371")
@pytest.mark.parametrize(
    "model_namespace, openvino_kserve_serving_runtime, ovms_kserve_inference_service",
    [
        pytest.param(
            {"name": "serverless-multi-models"},
            RunTimeConfigs.ONNX_OPSET13_RUNTIME_CONFIG,
            ONNX_SERVERLESS_INFERENCE_SERVICE_CONFIG,
        )
    ],
    indirect=True,
)
@pytest.mark.usefixtures("s3_mnist_serverless_inference_service")
class TestServerlessMultipleProjectsInNamespace:
    def test_serverless_multi_models_inference_bloom(
        self,
        ovms_kserve_inference_service,
    ):
        """Test inference with Bloom Caikit model when multiple models in the same namespace"""
        run_inference_multiple_times(
            isvc=ovms_kserve_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            run_in_parallel=True,
            iterations=5,
        )

    def test_serverless_multi_models_inference_flan(
        self, s3_mnist_serverless_inference_service, ovms_kserve_inference_service
    ):
        """Test inference with mnist model when multiple models in the same namespace"""
        run_inference_multiple_times(
            isvc=ovms_kserve_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            run_in_parallel=True,
            iterations=5,
        )
