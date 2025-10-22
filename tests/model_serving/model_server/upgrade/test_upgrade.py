import pytest

from tests.model_serving.model_server.upgrade.utils import (
    verify_inference_generation,
    verify_pod_containers_not_restarted,
    verify_serving_runtime_generation,
)
from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import ModelName, Protocols
from utilities.manifests.caikit_standalone import CAIKIT_STANDALONE_INFERENCE_CONFIG


TEST_RAW_CAIKIT_BGE_POST_UPGRADE_INFERENCE_EXISTS: str = "test_raw_caikit_bge_post_upgrade_inference_exists"


@pytest.mark.usefixtures("valid_aws_config")
class TestPreUpgradeModelServer:
    @pytest.mark.pre_upgrade
    @pytest.mark.rawdeployment
    def test_raw_caikit_bge_pre_upgrade_inference(self, caikit_raw_inference_service_scope_session):
        """Test Caikit bge-large-en embedding model inference using internal route before upgrade"""
        verify_inference_response(
            inference_service=caikit_raw_inference_service_scope_session,
            inference_config=CAIKIT_STANDALONE_INFERENCE_CONFIG,
            inference_type="embedding",
            protocol=Protocols.HTTP,
            model_name=ModelName.CAIKIT_BGE_LARGE_EN,
            use_default_query=True,
        )


class TestPostUpgradeModelServer:
    @pytest.mark.post_upgrade
    @pytest.mark.rawdeployment
    @pytest.mark.dependency(name=TEST_RAW_CAIKIT_BGE_POST_UPGRADE_INFERENCE_EXISTS)
    def test_raw_caikit_bge_post_upgrade_inference_exists(self, caikit_raw_inference_service_scope_session):
        """Test that raw deployment inference service exists after upgrade"""
        assert caikit_raw_inference_service_scope_session.exists

    @pytest.mark.post_upgrade
    @pytest.mark.rawdeployment
    @pytest.mark.dependency(depends=[TEST_RAW_CAIKIT_BGE_POST_UPGRADE_INFERENCE_EXISTS])
    def test_raw_caikit_bge_post_upgrade_inference_not_modified(self, caikit_raw_inference_service_scope_session):
        """Test that the raw deployment inference service is not modified in upgrade"""
        verify_inference_generation(isvc=caikit_raw_inference_service_scope_session, expected_generation=1)

    @pytest.mark.post_upgrade
    @pytest.mark.rawdeployment
    @pytest.mark.dependency(depends=[TEST_RAW_CAIKIT_BGE_POST_UPGRADE_INFERENCE_EXISTS])
    def test_raw_caikit_bge_post_upgrade_runtime_not_modified(self, caikit_raw_inference_service_scope_session):
        """Test that the raw deployment runtime is not modified in upgrade"""
        verify_serving_runtime_generation(isvc=caikit_raw_inference_service_scope_session, expected_generation=1)

    @pytest.mark.post_upgrade
    @pytest.mark.rawdeployment
    @pytest.mark.dependency(depends=[TEST_RAW_CAIKIT_BGE_POST_UPGRADE_INFERENCE_EXISTS])
    def test_raw_caikit_bge_post_upgrade_inference(self, caikit_raw_inference_service_scope_session):
        """Test Caikit bge-large-en embedding model inference using internal route after upgrade"""
        verify_inference_response(
            inference_service=caikit_raw_inference_service_scope_session,
            inference_config=CAIKIT_STANDALONE_INFERENCE_CONFIG,
            inference_type="embedding",
            protocol=Protocols.HTTP,
            model_name=ModelName.CAIKIT_BGE_LARGE_EN,
            use_default_query=True,
        )

    @pytest.mark.post_upgrade
    @pytest.mark.rawdeployment
    def test_verify_odh_model_controller_pod_not_restarted_post_upgrade(self, admin_client):
        """Verify that ODH Model Controller pod is not restarted after upgrade"""
        verify_pod_containers_not_restarted(
            client=admin_client,
            component_name="odh-model-controller",
        )

    @pytest.mark.post_upgrade
    @pytest.mark.rawdeployment
    def test_verify_kserve_pod_not_restarted_post_upgrade(self, admin_client):
        """Verify that KServe pod is not restarted after upgrade"""
        verify_pod_containers_not_restarted(
            client=admin_client,
            component_name="kserve",
        )
