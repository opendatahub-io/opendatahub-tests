import pytest
from ocp_resources.resource import ResourceEditor

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import ModelFormat, ModelStoragePath, Protocols, ModelInferenceRuntime
from utilities.constants import Labels
from utilities.inference_utils import Inference
from utilities.infra import check_pod_status_in_time, get_pods_by_isvc_label
from utilities.manifests.caikit_tgis import CAIKIT_TGIS_INFERENCE_CONFIG

pytestmark = pytest.mark.usefixtures("valid_aws_config")


@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri",
    [
        pytest.param(
            {"name": "kserve-raw-token-authentication"},
            {"model-dir": ModelStoragePath.FLAN_T5_SMALL_CAIKIT},
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

    @pytest.mark.sanity
    @pytest.mark.jira("RHOAIENG-19275", run=False)
    def test_raw_disable_enable_authentication_no_pod_rollout(self, http_s3_caikit_raw_inference_service):
        """Verify no pod rollout when disabling and enabling authentication"""
        pod = get_pods_by_isvc_label(
            client=http_s3_caikit_raw_inference_service.client,
            isvc=http_s3_caikit_raw_inference_service,
        )[0]

        ResourceEditor(
            patches={
                http_s3_caikit_raw_inference_service: {
                    "metadata": {
                        "labels": {Labels.KserveAuth.SECURITY: "false"},
                    }
                }
            }
        ).update()

        check_pod_status_in_time(pod=pod, status={pod.Status.RUNNING})

        ResourceEditor(
            patches={
                http_s3_caikit_raw_inference_service: {
                    "metadata": {
                        "labels": {Labels.KserveAuth.SECURITY: "true"},
                    }
                }
            }
        ).update()

        check_pod_status_in_time(pod=pod, status={pod.Status.RUNNING})

    @pytest.mark.dependency(depends=["test_disabled_raw_model_authentication"])
    def test_re_enabled_raw_model_authentication(self, http_s3_caikit_raw_inference_service, http_raw_inference_token):
        """Verify model query after authentication is re-enabled"""
        verify_inference_response(
            inference_service=http_s3_caikit_raw_inference_service,
            inference_config=CAIKIT_TGIS_INFERENCE_CONFIG,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.HTTPS,
            model_name=ModelFormat.CAIKIT,
            use_default_query=True,
            token=http_raw_inference_token,
        )

    @pytest.mark.skip(reason="Skipping test_cross_model_authentication_raw: RHOAIENG-19645")
    @pytest.mark.dependency(name="test_cross_model_authentication_raw")
    def test_cross_model_authentication_raw(self, http_s3_caikit_raw_inference_service_2, http_raw_inference_token):
        """Verify model with another model token"""

        verify_inference_response(
            inference_service=http_s3_caikit_raw_inference_service_2,
            runtime=ModelInferenceRuntime.CAIKIT_TGIS_RUNTIME,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.HTTPS,
            model_name=ModelFormat.CAIKIT,
            use_default_query=True,
            token=http_raw_inference_token,
            authorized_user=False,
        )
