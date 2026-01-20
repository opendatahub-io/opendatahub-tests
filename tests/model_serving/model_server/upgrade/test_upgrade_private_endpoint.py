import pytest

from tests.model_serving.model_server.upgrade.utils import (
    verify_inference_generation,
    verify_isvc_internal_access,
    verify_isvc_pods_not_restarted,
    verify_no_external_route,
    verify_private_endpoint_url,
    verify_serving_runtime_generation,
)
from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import Protocols
from utilities.inference_utils import Inference
from utilities.manifests.openvino import OPENVINO_KSERVE_INFERENCE_CONFIG


pytestmark = [pytest.mark.rawdeployment, pytest.mark.usefixtures("valid_aws_config")]


class TestPreUpgradePrivateEndpoint:
    """Pre-upgrade tests for private endpoint (internal-only) model serving."""

    @pytest.mark.pre_upgrade
    @pytest.mark.dependency(name="private_endpoint_isvc_created")
    def test_private_endpoint_pre_upgrade_exists(self, private_endpoint_inference_service_fixture):
        """Verify private endpoint InferenceService exists before upgrade"""
        assert private_endpoint_inference_service_fixture.exists, (
            f"InferenceService {private_endpoint_inference_service_fixture.name} does not exist"
        )

    @pytest.mark.pre_upgrade
    @pytest.mark.dependency(depends=["private_endpoint_isvc_created"])
    def test_private_endpoint_pre_upgrade_internal_url(self, private_endpoint_inference_service_fixture):
        """Verify InferenceService has internal cluster URL format before upgrade"""
        verify_private_endpoint_url(isvc=private_endpoint_inference_service_fixture)

    @pytest.mark.pre_upgrade
    @pytest.mark.dependency(depends=["private_endpoint_isvc_created"])
    def test_private_endpoint_pre_upgrade_no_external_route(
        self,
        admin_client,
        private_endpoint_inference_service_fixture,
    ):
        """Verify no external Route exists for private InferenceService before upgrade"""
        verify_no_external_route(client=admin_client, isvc=private_endpoint_inference_service_fixture)

    @pytest.mark.pre_upgrade
    @pytest.mark.dependency(depends=["private_endpoint_isvc_created"])
    def test_private_endpoint_pre_upgrade_inference(self, private_endpoint_inference_service_fixture):
        """Verify internal model inference before upgrade"""
        verify_inference_response(
            inference_service=private_endpoint_inference_service_fixture,
            inference_config=OPENVINO_KSERVE_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTP,
            use_default_query=True,
        )


class TestPostUpgradePrivateEndpoint:
    """Post-upgrade tests for private endpoint (internal-only) model serving."""

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="private_endpoint_post_exists")
    def test_private_endpoint_post_upgrade_exists(self, private_endpoint_inference_service_fixture):
        """Verify private endpoint InferenceService exists after upgrade"""
        assert private_endpoint_inference_service_fixture.exists, (
            f"InferenceService {private_endpoint_inference_service_fixture.name} does not exist after upgrade"
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["private_endpoint_post_exists"])
    def test_private_endpoint_post_upgrade_internal_url_preserved(self, private_endpoint_inference_service_fixture):
        """Verify internal cluster URL format is preserved after upgrade"""
        verify_private_endpoint_url(isvc=private_endpoint_inference_service_fixture)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["private_endpoint_post_exists"])
    def test_private_endpoint_post_upgrade_still_no_external_route(
        self,
        admin_client,
        private_endpoint_inference_service_fixture,
    ):
        """Verify no external Route was accidentally created during upgrade"""
        verify_no_external_route(client=admin_client, isvc=private_endpoint_inference_service_fixture)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["private_endpoint_post_exists"])
    def test_private_endpoint_post_upgrade_not_modified(self, private_endpoint_inference_service_fixture):
        """Verify InferenceService is not modified during upgrade"""
        verify_inference_generation(isvc=private_endpoint_inference_service_fixture, expected_generation=1)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["private_endpoint_post_exists"])
    def test_private_endpoint_post_upgrade_runtime_not_modified(self, private_endpoint_inference_service_fixture):
        """Verify ServingRuntime is not modified during upgrade"""
        verify_serving_runtime_generation(isvc=private_endpoint_inference_service_fixture, expected_generation=1)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["private_endpoint_post_exists"])
    def test_private_endpoint_post_upgrade_inference(self, private_endpoint_inference_service_fixture):
        """Verify internal model inference after upgrade"""
        verify_inference_response(
            inference_service=private_endpoint_inference_service_fixture,
            inference_config=OPENVINO_KSERVE_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTP,
            use_default_query=True,
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["private_endpoint_post_exists"])
    def test_private_endpoint_post_upgrade_internal_url_accessible(self, private_endpoint_inference_service_fixture):
        """Verify internal URL is accessible and properly formatted after upgrade"""
        internal_url = verify_isvc_internal_access(isvc=private_endpoint_inference_service_fixture)
        assert "svc.cluster.local" in internal_url, (
            f"Internal URL '{internal_url}' does not contain 'svc.cluster.local'"
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["private_endpoint_post_exists"])
    def test_private_endpoint_post_upgrade_pods_not_restarted(
        self,
        admin_client,
        private_endpoint_inference_service_fixture,
    ):
        """Verify InferenceService pods have not restarted during upgrade"""
        verify_isvc_pods_not_restarted(
            client=admin_client,
            isvc=private_endpoint_inference_service_fixture,
        )
