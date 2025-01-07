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

pytestmark = pytest.mark.usefixtures("valid_aws_config")


@pytest.mark.raw_deployment
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, serving_runtime_from_template, s3_models_inference_service",
    [
        pytest.param(
            {"name": "raw-deployment-caikit-flan"},
            {"model-dir": ModelStoragePath.FLAN_T5_SMALL},
            {
                "name": f"{Protocols.HTTP}-{ModelInferenceRuntime.CAIKIT_TGIS_RUNTIME}",
                "template-name": RuntimeTemplates.CAIKIT_TGIS_SERVING,
                "multi-model": False,
                "enable-http": True,
            },
            {"name": f"{Protocols.HTTP}-{ModelFormat.CAIKIT}", "deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT},
        )
    ],
    indirect=True,
)
class TestRestRawDeployment:
    def test_default_visibility_value(self, s3_models_inference_service):
        """Test default route visibility value"""
        assert s3_models_inference_service.annotations.get("networking.kserve.io/visibility") is None

    def test_rest_raw_deployment_internal_route(self, s3_models_inference_service):
        """Test HTTP inference using internal route"""
        verify_inference_response(
            inference_service=s3_models_inference_service,
            runtime=ModelInferenceRuntime.CAIKIT_TGIS_RUNTIME,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.HTTP,
            model_name=ModelFormat.CAIKIT,
            use_default_query=True,
        )

    @pytest.mark.jira("RHOAIENG-17322", run=False)
    @pytest.mark.parametrize(
        "patched_http_s3_caikit_raw_isvc_visibility_label",
        [
            pytest.param(
                {"visibility": "exposed"},
            )
        ],
        indirect=True,
    )
    @pytest.mark.dependency(name="test_rest_raw_deployment_exposed_route")
    def test_rest_raw_deployment_exposed_route(self, patched_http_s3_caikit_raw_isvc_visibility_label):
        """Test HTTP inference using exposed (external) route"""
        verify_inference_response(
            inference_service=patched_http_s3_caikit_raw_isvc_visibility_label,
            runtime=ModelInferenceRuntime.CAIKIT_TGIS_RUNTIME,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.HTTPS,
            model_name=ModelFormat.CAIKIT,
            use_default_query=True,
        )

    @pytest.mark.dependency(depends=["test_rest_raw_deployment_exposed_route"])
    @pytest.mark.parametrize(
        "patched_http_s3_caikit_raw_isvc_visibility_label",
        [
            pytest.param(
                {"visibility": "local-cluster"},
            )
        ],
        indirect=True,
    )
    def test_disabled_rest_raw_deployment_exposed_route(self, patched_http_s3_caikit_raw_isvc_visibility_label):
        """Test HTTP inference fails when using external route after it was disabled"""
        verify_inference_response(
            inference_service=patched_http_s3_caikit_raw_isvc_visibility_label,
            runtime=ModelInferenceRuntime.CAIKIT_TGIS_RUNTIME,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.HTTP,
            model_name=ModelFormat.CAIKIT,
            use_default_query=True,
        )
