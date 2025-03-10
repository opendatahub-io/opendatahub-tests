import pytest

from tests.model_serving.model_server.serverless.utils import verify_canary_traffic
from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import (
    KServeDeploymentType,
    ModelAndFormat,
    ModelName,
    ModelStoragePath,
    Protocols,
    RuntimeTemplates,
)
from utilities.inference_utils import Inference
from utilities.manifests.pytorch import PYTORCH_TGIS_INFERENCE_CONFIG
from utilities.manifests.tgis_grpc import TGIS_INFERENCE_CONFIG

pytestmark = [pytest.mark.serverless, pytest.mark.sanity]


@pytest.mark.polarion("ODS-2371")
@pytest.mark.parametrize(
    "model_namespace, serving_runtime_from_template, s3_models_inference_service",
    [
        pytest.param(
            {"name": "serverless-canary-rollout"},
            {
                "name": "tgis-runtime",
                "template-name": RuntimeTemplates.TGIS_GRPC_SERVING,
                "multi-model": False,
                "enable-http": False,
                "enable-grpc": True,
            },
            {
                "name": f"{ModelName.BLOOM_560M}-model",
                "deployment-mode": KServeDeploymentType.SERVERLESS,
                "model-dir": f"{ModelStoragePath.BLOOM_560M_CAIKIT}/artifacts",
                "external-route": True,
            },
        )
    ],
    indirect=True,
)
class TestServerlessCanaryRollout:
    def test_serverless_before_model_update(
        self,
        s3_models_inference_service,
    ):
        """Test inference with Bloom before model is updated."""
        verify_inference_response(
            inference_service=s3_models_inference_service,
            inference_config=PYTORCH_TGIS_INFERENCE_CONFIG,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.GRPC,
            model_name=ModelAndFormat.BLOOM_560M_CAIKIT,
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
