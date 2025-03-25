import pytest

from tests.model_serving.model_server.serverless.utils import verify_canary_traffic
from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import (
    ModelAndFormat,
    ModelFormat,
    ModelInferenceRuntime,
    ModelStoragePath,
    ModelVersion,
    Protocols,
)
from utilities.inference_utils import Inference
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG
from utilities.manifests.tgis_grpc import TGIS_INFERENCE_CONFIG

pytestmark = [pytest.mark.serverless, pytest.mark.sanity]


@pytest.mark.polarion("ODS-2371")
@pytest.mark.parametrize(
    "model_namespace, openvino_kserve_serving_runtime, ovms_serverless_inference_service",
    [
        pytest.param(
            {"name": "serverless-canary-rollout"},
            {
                "runtime-name": ModelInferenceRuntime.ONNX_RUNTIME,
                "model-format": {ModelFormat.ONNX: ModelVersion.OPSET13},
            },
            {
                "name": {ModelFormat.ONNX},
                "model-version": ModelVersion.OPSET13,
                "model-dir": "test-dir",
                # "deployment-mode": KServeDeploymentType.SERVERLESS,
            },
        )
    ],
    indirect=True,
)
class TestServerlessCanaryRollout:
    def test_serverless_before_model_update(
        self,
        ovms_serverless_inference_service,
    ):
        """Test inference before model is updated."""
        verify_inference_response(
            inference_service=ovms_serverless_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )

    @pytest.mark.parametrize(
        "inference_service_updated_canary_config",
        [
            pytest.param(
                {"canary-traffic-percent": 30, "model-path": ModelStoragePath.FLAN_T5_SMALL_HF},
            )
        ],
        indirect=True,
    )
    def test_serverless_during_canary_rollout(self, inference_service_updated_canary_config):
        """Test inference during canary rollout"""
        verify_canary_traffic(
            isvc=inference_service_updated_canary_config,
            inference_config=TGIS_INFERENCE_CONFIG,
            model_name=ModelAndFormat.FLAN_T5_SMALL_CAIKIT,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.GRPC,
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
            inference_config=TGIS_INFERENCE_CONFIG,
            model_name=ModelAndFormat.FLAN_T5_SMALL_CAIKIT,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.GRPC,
            iterations=5,
            expected_percentage=100,
        )
