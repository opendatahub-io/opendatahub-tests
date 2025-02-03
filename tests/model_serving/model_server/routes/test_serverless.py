import pytest

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import (
    KServeDeploymentType,
    ModelFormat,
    ModelStoragePath,
    Protocols,
    ModelInferenceRuntime,
    RuntimeTemplates,
)
from utilities.inference_utils import Inference
from utilities.manifests.caikit_tgis import CAIKIT_TGIS_INFERENCE_CONFIG

pytestmark = [pytest.mark.usefixtures("valid_aws_config"), pytest.mark.serverless, pytest.mark.sanity]


@pytest.mark.parametrize(
    "model_namespace, serving_runtime_from_template, s3_models_inference_service",
    [
        pytest.param(
            {"name": "serverless-rcaikit-routes"},
            {
                "name": f"{Protocols.HTTP}-{ModelInferenceRuntime.CAIKIT_TGIS_RUNTIME}",
                "template-name": RuntimeTemplates.CAIKIT_TGIS_SERVING,
                "multi-model": False,
                "enable-http": True,
            },
            {
                "name": f"{Protocols.HTTP}-{ModelFormat.CAIKIT}",
                "deployment-mode": KServeDeploymentType.SERVERLESS,
                "model-dir": ModelStoragePath.FLAN_T5_SMALL,
            },
        )
    ],
    indirect=True,
)
class TestRestServerlessRoutes:
    def test_serverless_default_visibility_value(self, s3_models_inference_service):
        """Test default route visibility value"""
        if labels := s3_models_inference_service.labels:
            assert labels.get("networking.kserve.io/visibility") is None

    def test_rest_serverless_external_route(self, s3_models_inference_service):
        """Test HTTP inference using internal route"""
        verify_inference_response(
            inference_service=s3_models_inference_service,
            inference_config=CAIKIT_TGIS_INFERENCE_CONFIG,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.HTTPS,
            model_name=ModelFormat.CAIKIT,
            use_default_query=True,
        )

    @pytest.mark.parametrize(
        "patched_s3_caikit_kserve_isvc_visibility_label",
        [
            pytest.param(
                {"visibility": "local-cluster"},
            )
        ],
        indirect=True,
    )
    @pytest.mark.dependency(name="test_rest_serverless_internal_route")
    def test_rest_serverless_internal_route(self, patched_s3_caikit_kserve_isvc_visibility_label):
        """Test HTTP inference using exposed (external) route"""
        verify_inference_response(
            inference_service=patched_s3_caikit_kserve_isvc_visibility_label,
            inference_config=CAIKIT_TGIS_INFERENCE_CONFIG,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.HTTPS,
            model_name=ModelFormat.CAIKIT,
            use_default_query=True,
        )

    @pytest.mark.dependency(depends=["test_rest_serverless_internal_route"])
    @pytest.mark.parametrize(
        "patched_s3_caikit_kserve_isvc_visibility_label",
        [
            pytest.param(
                {"visibility": "exposed"},
            )
        ],
        indirect=True,
    )
    def test_rest_serverless_exposed_label_route(self, patched_s3_caikit_kserve_isvc_visibility_label):
        """Test HTTP inference fails when using external route after it was disabled"""
        verify_inference_response(
            inference_service=patched_s3_caikit_kserve_isvc_visibility_label,
            inference_config=CAIKIT_TGIS_INFERENCE_CONFIG,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.HTTPS,
            model_name=ModelFormat.CAIKIT,
            use_default_query=True,
        )
