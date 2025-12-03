import pytest
from ocp_resources.resource import ResourceEditor

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import Annotations, Protocols
from utilities.inference_utils import Inference
from utilities.infra import check_pod_status_in_time, get_pods_by_isvc_label
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG

pytestmark = [pytest.mark.serverless, pytest.mark.usefixtures("valid_aws_config")]


@pytest.mark.parametrize(
    "unprivileged_model_namespace",
    [
        pytest.param(
            {"name": "kserve-token-authentication"},
        )
    ],
    indirect=True,
)
class TestKserveServerlessTokenAuthentication:
    @pytest.mark.smoke
    @pytest.mark.ocp_interop
    @pytest.mark.dependency(name="test_model_authentication_using_rest")
    def test_model_authentication_using_rest(self, http_ovms_serverless_inference_service, http_ovms_inference_token):
        """Verify model query with token using REST"""
        verify_inference_response(
            inference_service=http_ovms_serverless_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
            token=http_ovms_inference_token,
        )

    @pytest.mark.dependency(name="test_disabled_model_authentication")
    def test_disabled_model_authentication(self, patched_remove_ovms_authentication_isvc):
        """Verify model query after authentication is disabled"""
        verify_inference_response(
            inference_service=patched_remove_ovms_authentication_isvc,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )

    @pytest.mark.dependency(depends=["test_disabled_model_authentication"])
    def test_re_enabled_model_authentication(self, http_ovms_serverless_inference_service, http_ovms_inference_token):
        """Verify model query after authentication is re-enabled"""
        verify_inference_response(
            inference_service=http_ovms_serverless_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
            token=http_ovms_inference_token,
        )

    def test_model_authentication_using_invalid_token(self, http_ovms_serverless_inference_service):
        """Verify model query with an invalid token"""
        verify_inference_response(
            inference_service=http_ovms_serverless_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
            token="dummy",
            authorized_user=False,
        )

    def test_model_authentication_without_token(self, http_ovms_serverless_inference_service):
        """Verify model query without providing a token"""
        verify_inference_response(
            inference_service=http_ovms_serverless_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
            authorized_user=False,
        )

    @pytest.mark.sanity
    def test_serverless_disable_enable_authentication_no_pod_rollout(self, http_ovms_serverless_inference_service):
        """Verify no pod rollout when disabling and enabling authentication"""
        pod = get_pods_by_isvc_label(
            client=http_ovms_serverless_inference_service.client,
            isvc=http_ovms_serverless_inference_service,
        )[0]

        ResourceEditor(
            patches={
                http_ovms_serverless_inference_service: {
                    "metadata": {
                        "annotations": {Annotations.KserveAuth.SECURITY: "false"},
                    }
                }
            }
        ).update()

        check_pod_status_in_time(pod=pod, status={pod.Status.RUNNING})

        ResourceEditor(
            patches={
                http_ovms_serverless_inference_service: {
                    "metadata": {
                        "annotations": {Annotations.KserveAuth.SECURITY: "true"},
                    }
                }
            }
        ).update()

        check_pod_status_in_time(pod=pod, status={pod.Status.RUNNING})

