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


@pytest.mark.parametrize(
    "ns_with_modelmesh_enabled, http_s3_openvino_model_mesh_inference_service",
    [
        pytest.param(
            {"name": "model-mesh-openvino"},
            {"model-path": ModelStoragePath.OPENVINO_EXAMPLE_MODEL},
        )
    ],
    indirect=True,
)
class TestOpenVINO:
    @pytest.mark.smoke
    @pytest.mark.polarion("ODS-2053", "ODS-2054")
    def test_model_mesh_openvino_rest_inference(self, http_s3_openvino_model_mesh_inference_service):
        verify_inference_response(
            inference_service=http_s3_openvino_model_mesh_inference_service,
            runtime=RuntimeQueryKeys.OPENVINO_RUNTIME,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTP,
            model_name=ModelFormat.OPENVINO,
            use_default_query=True,
        )

    @pytest.mark.sanity
    @pytest.mark.skip(reason="product bug?")
    @pytest.mark.polarion("ODS-1920")
    def test_model_mesh_openvino_inference_with_token(
        self,
        patched_model_mesh_sr_with_authentication,
        http_s3_openvino_model_mesh_inference_service,
        model_mesh_inference_token,
    ):
        verify_inference_response(
            inference_service=http_s3_openvino_model_mesh_inference_service,
            runtime=RuntimeQueryKeys.OPENVINO_RUNTIME,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTP,
            model_name=ModelFormat.OPENVINO,
            use_default_query=True,
            token=model_mesh_inference_token,
        )
