import pytest

from tests.model_serving.model_server.authentication.utils import (
    verify_inference_response,
)
from utilities.constants import (
    ModelFormat,
    ModelStoragePath,
    Protocols,
    RuntimeQueryKeys,
)
from utilities.inference_utils import Inference

pytestmark = pytest.mark.usefixtures("valid_aws_config")


@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri",
    [
        pytest.param(
            {"name": "raw-deployment"},
            {"model-dir": ModelStoragePath.FLAN_T5_SMALL},
        )
    ],
    indirect=True,
)
class TestRestRawDeployment:
    def test_default_visibility_value(self, http_s3_caikit_raw_inference_service):
        """Test default route visibility value"""
        assert http_s3_caikit_raw_inference_service.annotations.get("networking.kserve.io/visibility") is None

    def test_rest_raw_deployment_internal_route(self, http_s3_caikit_raw_inference_service):
        """Test HTTP inference using internal route"""
        verify_inference_response(
            inference_service=http_s3_caikit_raw_inference_service,
            runtime=RuntimeQueryKeys.CAIKIT_TGIS_RUNTIME,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.HTTP,
            model_name=ModelFormat.CAIKIT,
            use_default_query=True,
        )

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
            runtime=RuntimeQueryKeys.CAIKIT_TGIS_RUNTIME,
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
            runtime=RuntimeQueryKeys.CAIKIT_TGIS_RUNTIME,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.HTTP,
            model_name=ModelFormat.CAIKIT,
            use_default_query=True,
        )
