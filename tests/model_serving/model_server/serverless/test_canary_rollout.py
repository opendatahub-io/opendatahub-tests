import pytest

from tests.model_serving.model_server.serverless.constants import (
    ONNX_SERVERLESS_INFERENCE_SERVICE_CONFIG,
)
from tests.model_serving.model_server.serverless.utils import verify_canary_traffic
from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import (
    ModelStoragePath,
    Protocols,
    RunTimeConfigs,
)
from utilities.inference_utils import Inference
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG
from utilities.manifests.openvino import OPENVINO_INFERENCE_CONFIG

pytestmark = [pytest.mark.serverless, pytest.mark.sanity]


@pytest.mark.polarion("ODS-2371")
@pytest.mark.parametrize(
    "model_namespace, ovms_kserve_serving_runtime, ovms_kserve_inference_service",
    [
        pytest.param(
            {"name": "serverless-canary-rollout"},
            RunTimeConfigs.ONNX_OPSET13_RUNTIME_CONFIG,
            ONNX_SERVERLESS_INFERENCE_SERVICE_CONFIG,
        )
    ],
    indirect=True,
)
class TestServerlessCanaryRollout:
    def test_serverless_before_model_update(
        self,
        ovms_kserve_inference_service,
    ):
        """Test inference before model is updated."""
        verify_inference_response(
            inference_service=ovms_kserve_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )

    @pytest.mark.parametrize(
        "inference_service_updated_canary_config",
        [
            pytest.param(
                {"canary-traffic-percent": 30, "model-path": ModelStoragePath.MNIST_8_ONNX},
            )
        ],
        indirect=True,
    )
    def test_serverless_during_canary_rollout(self, inference_service_updated_canary_config):
        """Test inference during canary rollout"""
        verify_canary_traffic(
            isvc=inference_service_updated_canary_config,
            inference_config=OPENVINO_INFERENCE_CONFIG,
            inference_type=Inference.MNIST,
            protocol=Protocols.HTTPS,
            iterations=20,
            expected_percentage=30,
            tolerance=10,
        )

    @pytest.mark.parametrize(
        "inference_service_updated_canary_config",
        [
            pytest.param(
                {"canary-traffic-percent": 100},
            )
        ],
        indirect=True,
    )
    def test_serverless_after_canary_rollout(self, inference_service_updated_canary_config):
        """Test inference after canary rollout"""
        verify_canary_traffic(
            isvc=inference_service_updated_canary_config,
            inference_config=OPENVINO_INFERENCE_CONFIG,
            inference_type=Inference.MNIST,
            protocol=Protocols.HTTPS,
            iterations=5,
            expected_percentage=100,
        )
