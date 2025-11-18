import pytest

from utilities.manifests.openvino import OPENVINO_KSERVE_INFERENCE_CONFIG
from tests.model_serving.model_server.upgrade.utils import (
    verify_inference_generation,
    verify_pod_containers_not_restarted,
    verify_serving_runtime_generation,
)
from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import Protocols


pytestmark = [pytest.mark.rawdeployment, pytest.mark.usefixtures("valid_aws_config")]


class TestPreUpgradeModelServer:
    @pytest.mark.pre_upgrade
    def test_raw_deployment_pre_upgrade_inference(self, inference_service_fixture):
        """Test raw deployment model inference using internal route before upgrade"""
        verify_inference_response(
            inference_service=inference_service_fixture,
            inference_config=OPENVINO_KSERVE_INFERENCE_CONFIG,
            inference_type="infer",
            protocol=Protocols.HTTP,
            use_default_query=True,
        )


class TestPostUpgradeModelServer:
    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="isvc_exists")
    def test_raw_deployment_post_upgrade_inference_exists(self, inference_service_fixture):
        """Test that raw deployment inference service exists after upgrade"""
        assert inference_service_fixture.exists

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["isvc_exists"])
    def test_raw_deployment_post_upgrade_inference_not_modified(self, inference_service_fixture):
        """Test that the raw deployment inference service is not modified in upgrade"""
        verify_inference_generation(isvc=inference_service_fixture, expected_generation=1)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["isvc_exists"])
    def test_raw_deployment_post_upgrade_runtime_not_modified(self, inference_service_fixture):
        """Test that the raw deployment runtime is not modified in upgrade"""
        verify_serving_runtime_generation(isvc=inference_service_fixture, expected_generation=1)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["isvc_exists"])
    def test_raw_deployment_post_upgrade_inference(self, inference_service_fixture):
        """Test raw deployment model inference using internal route after upgrade"""
        verify_inference_response(
            inference_service=inference_service_fixture,
            inference_config=OPENVINO_KSERVE_INFERENCE_CONFIG,
            inference_type="infer",
            protocol=Protocols.HTTP,
            use_default_query=True,
        )

    @pytest.mark.post_upgrade
    def test_verify_odh_model_controller_pod_not_restarted_post_upgrade(self, admin_client):
        """Verify that ODH Model Controller pod is not restarted after upgrade"""
        verify_pod_containers_not_restarted(
            client=admin_client,
            component_name="odh-model-controller",
        )

    @pytest.mark.post_upgrade
    def test_verify_kserve_pod_not_restarted_post_upgrade(self, admin_client):
        """Verify that KServe pod is not restarted after upgrade"""
        verify_pod_containers_not_restarted(
            client=admin_client,
            component_name="kserve",
        )
