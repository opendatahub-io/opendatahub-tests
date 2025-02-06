import pytest

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import ModelFormat, ModelStoragePath, Protocols
from utilities.inference_utils import Inference
from utilities.manifests.caikit_tgis import CAIKIT_TGIS_INFERENCE_CONFIG

pytestmark = pytest.mark.usefixtures("valid_aws_config")


@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri",
    [
        pytest.param(
            {"name": "kserve-raw-token-authentication"},
            {"model-dir": ModelStoragePath.FLAN_T5_SMALL},
        )
    ],
    indirect=True,
)
class TestKserveTokenAuthenticationRawForRest:
    @pytest.mark.smoke
    @pytest.mark.dependency(name="test_model_authentication_using_rest_raw")
    def test_model_authentication_using_rest_raw(self, http_s3_caikit_raw_inference_service, http_raw_inference_token):
        """Verify RAW Kserve model query with token using REST"""
        verify_inference_response(
            inference_service=http_s3_caikit_raw_inference_service,
            inference_config=CAIKIT_TGIS_INFERENCE_CONFIG,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.HTTPS,
            model_name=ModelFormat.CAIKIT,
            use_default_query=True,
            token=http_raw_inference_token,
        )

    @pytest.mark.dependency(name="test_disabled_raw_model_authentication")
    def test_disabled_raw_model_authentication(self, patched_remove_raw_authentication_isvc):
        """Verify model query after authentication is disabled"""
        verify_inference_response(
            inference_service=patched_remove_raw_authentication_isvc,
            inference_config=CAIKIT_TGIS_INFERENCE_CONFIG,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.HTTPS,
            model_name=ModelFormat.CAIKIT,
            use_default_query=True,
        )
