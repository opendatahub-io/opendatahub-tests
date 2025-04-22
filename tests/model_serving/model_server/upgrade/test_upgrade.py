import pytest

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import ModelName, Protocols
from utilities.inference_utils import Inference
from utilities.manifests.caikit_standalone import CAIKIT_STANDALONE_INFERENCE_CONFIG
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG
from utilities.manifests.openvino import OPENVINO_INFERENCE_CONFIG


# TODO: add auth, external route and grpc tests


@pytest.mark.usefixtures("valid_aws_config")
class TestPreUpgradeModelServer:
    @pytest.mark.pre_upgrade
    @pytest.mark.serverless
    def test_serverless_onnx_pre_upgrade_inference(self, ovms_serverless_inference_service_scope_session):
        """Verify that kserve Serverless ONNX model can be queried using REST before upgrade"""
        verify_inference_response(
            inference_service=ovms_serverless_inference_service_scope_session,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )

    @pytest.mark.pre_upgrade
    @pytest.mark.rawdeployment
    def test_raw_caikit_bge_pre_upgrade_inference(self, caikit_raw_inference_service_scope_session):
        """Test Caikit bge-large-en embedding model inference using internal route before upgrade"""
        verify_inference_response(
            inference_service=caikit_raw_inference_service_scope_session,
            inference_config=CAIKIT_STANDALONE_INFERENCE_CONFIG,
            inference_type="embedding",
            protocol=Protocols.HTTPS,
            model_name=ModelName.CAIKIT_BGE_LARGE_EN,
            use_default_query=True,
        )

    @pytest.mark.pre_upgrade
    @pytest.mark.modelmesh
    def test_model_mesh_openvino_pre_upgrade_inference(self, openvino_model_mesh_inference_service_scope_session):
        """Test OpenVINO ModelMesh inference with internal route before upgrade"""
        verify_inference_response(
            inference_service=openvino_model_mesh_inference_service_scope_session,
            inference_config=OPENVINO_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTP,
            use_default_query=True,
        )

    @pytest.mark.pre_upgrade
    @pytest.mark.serverless
    def test_serverless_authenticated_onnx_pre_upgrade_inference(
        self, ovms_authenticated_serverless_inference_service_scope_session, http_inference_token_scope_session
    ):
        """Verify that kserve Serverless with auth ONNX model can be queried using REST before upgrade"""
        verify_inference_response(
            inference_service=ovms_authenticated_serverless_inference_service_scope_session,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            token=http_inference_token_scope_session,
            use_default_query=True,
        )


class TestPostUpgradeModelServer:
    @pytest.mark.post_upgrade
    @pytest.mark.serverless
    @pytest.mark.dependency(name="test_serverless_onnx_post_upgrade_inference_service_exists")
    def test_serverless_onnx_post_upgrade_inference_service_exists(
        self, ovms_serverless_inference_service_scope_session
    ):
        """Test that the serverless inference service exists after upgrade"""
        assert ovms_serverless_inference_service_scope_session.exists

    @pytest.mark.post_upgrade
    @pytest.mark.serverless
    @pytest.mark.dependency(depends=["test_serverless_onnx_post_upgrade_inference_service_exists"])
    def test_serverless_onnx_post_upgrade_inference(self, ovms_serverless_inference_service_scope_session):
        """Verify that kserve Serverless ONNX model can be queried using REST after upgrade"""
        verify_inference_response(
            inference_service=ovms_serverless_inference_service_scope_session,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )

    @pytest.mark.post_upgrade
    @pytest.mark.rawdeployment
    @pytest.mark.dependency(name="test_raw_caikit_bge_post_upgrade_inference_exists")
    def test_raw_caikit_bge_post_upgrade_inference_exists(self, caikit_raw_inference_service_scope_session):
        """Test that raw deployment inference service exists after upgrade"""
        assert caikit_raw_inference_service_scope_session.exists

    @pytest.mark.post_upgrade
    @pytest.mark.rawdeployment
    @pytest.mark.dependency(depends=["test_raw_caikit_bge_post_upgrade_inference_exists"])
    def test_raw_caikit_bge_post_upgrade_inference(self, caikit_raw_inference_service_scope_session):
        """Test Caikit bge-large-en embedding model inference using internal route after upgrade"""
        verify_inference_response(
            inference_service=caikit_raw_inference_service_scope_session,
            inference_config=CAIKIT_STANDALONE_INFERENCE_CONFIG,
            inference_type="embedding",
            protocol=Protocols.HTTPS,
            model_name=ModelName.CAIKIT_BGE_LARGE_EN,
            use_default_query=True,
        )

    @pytest.mark.post_upgrade
    @pytest.mark.modelmesh
    @pytest.mark.dependency(name="test_model_mesh_openvino_post_upgrade_inference_exists")
    def test_model_mesh_openvino_post_upgrade_inference_exists(
        self, openvino_model_mesh_inference_service_scope_session
    ):
        """Test that model mesh inference service exists after upgrade"""
        assert openvino_model_mesh_inference_service_scope_session.exists

    @pytest.mark.post_upgrade
    @pytest.mark.modelmesh
    @pytest.mark.dependency(depends=["test_model_mesh_openvino_post_upgrade_inference_exists"])
    def test_model_mesh_openvino_post_upgrade_inference(self, openvino_model_mesh_inference_service_scope_session):
        """Test OpenVINO ModelMesh inference with internal route after upgrade"""
        verify_inference_response(
            inference_service=openvino_model_mesh_inference_service_scope_session,
            inference_config=OPENVINO_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTP,
            use_default_query=True,
        )

    @pytest.mark.post_upgrade
    @pytest.mark.serverless
    @pytest.mark.dependency(name="test_serverless_authenticated_onnx_post_upgrade_inference_service_exists")
    def test_serverless_authenticated_onnx_post_upgrade_inference_service_exists(
        self, ovms_authenticated_serverless_inference_service_scope_session
    ):
        """Test that the serverless inference service exists after upgrade"""
        assert ovms_authenticated_serverless_inference_service_scope_session.exists

    @pytest.mark.post_upgrade
    @pytest.mark.serverless
    @pytest.mark.dependency(depends=["test_serverless_authenticated_onnx_post_upgrade_inference_service_exists"])
    def test_serverless_authenticated_onnx_post_upgrade_inference(
        self, ovms_authenticated_serverless_inference_service_scope_session, http_inference_token_scope_session
    ):
        """Verify that kserve Serverless with auth ONNX model can be queried using REST before upgrade"""
        verify_inference_response(
            inference_service=ovms_authenticated_serverless_inference_service_scope_session,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            token=http_inference_token_scope_session,
            use_default_query=True,
        )
