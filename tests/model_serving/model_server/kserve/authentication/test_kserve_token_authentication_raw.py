import pytest
import requests

from tests.model_serving.model_runtime.mlserver.utils import validate_deterministic_snapshot
from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import Protocols
from utilities.inference_utils import Inference, get_exposed_isvc_url

pytestmark = pytest.mark.usefixtures("valid_aws_config")


def assert_auth_inference(isvc, model_config, token=None, authorized=True):
    """Check auth behavior with the architecture's matching inference payload."""
    if "request" in model_config:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        response = requests.post(
            f"{get_exposed_isvc_url(isvc=isvc)}/v2/models/{isvc.name}/infer",
            json=model_config["request"],
            headers=headers,
            verify=False,
            timeout=60,
        )
        if authorized:
            response.raise_for_status()
            validate_deterministic_snapshot(response=response.json())
        else:
            assert response.status_code in (401, 403), response.text
        return

    verify_inference_response(
        inference_service=isvc,
        inference_config=model_config["inference"],
        inference_type=Inference.INFER,
        protocol=Protocols.HTTPS,
        use_default_query=True,
        token=token,
        authorized_user=authorized,
    )


@pytest.mark.smoke
@pytest.mark.rawdeployment
@pytest.mark.arch_runtime
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
    """Validate KServe raw deployment token-based authentication for REST inference.

    Steps:
        1. Deploy OVMS on x86 or MLServer on ARM64 with authentication enabled.
        2. Query the model with a valid token and verify a successful REST inference response.
        3. Disable authentication and verify the model is still queryable without a token.
        4. Re-enable authentication and verify the model requires a valid token again.
        5. Attempt cross-model authentication using another model's token and verify access is denied.
    """

    @pytest.mark.smoke
    @pytest.mark.ocp_interop
    @pytest.mark.dependency(name="test_model_authentication_using_rest_raw")
    def test_model_authentication_using_rest_raw(
        self, http_s3_ovms_raw_inference_service, http_raw_inference_token, auth_model_config
    ):
        """Verify RAW Kserve model query with token using REST"""
        assert_auth_inference(http_s3_ovms_raw_inference_service, auth_model_config, token=http_raw_inference_token)

    @pytest.mark.dependency(name="test_disabled_raw_model_authentication")
    def test_disabled_raw_model_authentication(self, patched_remove_raw_authentication_isvc, auth_model_config):
        """Verify model query after authentication is disabled"""
        assert_auth_inference(patched_remove_raw_authentication_isvc, auth_model_config)

    def test_re_enabled_raw_model_authentication(
        self, http_s3_ovms_raw_inference_service, http_raw_inference_token, auth_model_config
    ):
        """Verify model query after authentication is re-enabled"""
        assert_auth_inference(http_s3_ovms_raw_inference_service, auth_model_config, token=http_raw_inference_token)

    @pytest.mark.parametrize(
        "http_s3_ovms_raw_inference_service_2",
        [pytest.param({"model-dir": "test-dir"})],
        indirect=True,
    )
    @pytest.mark.dependency(name="test_cross_model_authentication_raw")
    def test_cross_model_authentication_raw(
        self, http_s3_ovms_raw_inference_service_2, http_raw_inference_token, auth_model_config
    ):
        """Verify model with another model token"""
        assert_auth_inference(
            http_s3_ovms_raw_inference_service_2,
            auth_model_config,
            token=http_raw_inference_token,
            authorized=False,
        )
