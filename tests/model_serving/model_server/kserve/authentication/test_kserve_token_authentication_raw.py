import pytest

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import Protocols
from utilities.inference_utils import Inference
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG

pytestmark = pytest.mark.usefixtures("valid_aws_config")


@pytest.mark.smoke
@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "unprivileged_model_namespace, http_s3_ovms_raw_inference_service",
    [
        pytest.param(
            {"name": "test-kserve-raw-token-authentication"},
            {"model-dir": "test-dir"},
        )
    ],
    indirect=True,
)
class TestKserveTokenAuthenticationRawForRest:
    @pytest.mark.smoke
    @pytest.mark.ocp_interop
    @pytest.mark.dependency(name="test_model_authentication_using_rest_raw")
    def test_model_authentication_using_rest_raw(self, http_s3_ovms_raw_inference_service, http_raw_inference_token):
        """Verify RAW Kserve model query with token using REST"""
        verify_inference_response(
            inference_service=http_s3_ovms_raw_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
            token=http_raw_inference_token,
        )

    @pytest.mark.dependency(name="test_disabled_raw_model_authentication")
    def test_disabled_raw_model_authentication(self, patched_remove_raw_authentication_isvc):
        """Verify model query after authentication is disabled"""
        verify_inference_response(
            inference_service=patched_remove_raw_authentication_isvc,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )

    def test_re_enabled_raw_model_authentication(self, http_s3_ovms_raw_inference_service, http_raw_inference_token):
        """Verify model query after authentication is re-enabled"""
        verify_inference_response(
            inference_service=http_s3_ovms_raw_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
            token=http_raw_inference_token,
        )

    @pytest.mark.parametrize(
        "http_s3_ovms_raw_inference_service_2",
        [pytest.param({"model-dir": "test-dir"})],
        indirect=True,
    )
    @pytest.mark.dependency(name="test_cross_model_authentication_raw")
    def test_cross_model_authentication_raw(self, http_s3_ovms_raw_inference_service_2, http_raw_inference_token):
        """Verify model with another model token"""
        verify_inference_response(
            inference_service=http_s3_ovms_raw_inference_service_2,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
            token=http_raw_inference_token,
            authorized_user=False,
        )
