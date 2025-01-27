import pytest

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import (
    ModelStoragePath,
    Protocols,
    ModelInferenceRuntime,
)
from utilities.inference_utils import Inference


pytestmark = [pytest.mark.modelmesh]


@pytest.mark.parametrize(
    "model_namespace, http_s3_openvino_model_mesh_inference_service, http_s3_tensorflow_model_mesh_inference_service",
    [
        pytest.param(
            {"name": "model-mesh-openvino", "modelmesh-enabled": True},
            {"enable-external-route": True},
            {"model-path": ModelStoragePath.OPENVINO_EXAMPLE_MODEL},
            {"enable-external-route": True},
            {"model-path": ModelStoragePath.TENSORFLOW_MODEL},
        )
    ],
    indirect=True,
)
class TestOpenVINOModelMeshMultiServers:
    def test_model_mesh_openvino_rest_inference_internal_route(
        self,
        http_s3_openvino_model_mesh_inference_service,
        http_s3_tensorflow_model_mesh_inference_service,
    ):
        verify_inference_response(
            inference_service=http_s3_openvino_model_mesh_inference_service,
            runtime=ModelInferenceRuntime.OPENVINO_RUNTIME,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTP,
            use_default_query=True,
        )

    def test_model_mesh_tensorflow_rest_inference_external_route(
        self,
        http_s3_tensorflow_model_mesh_inference_service,
        http_s3_openvino_model_mesh_inference_service,
    ):
        verify_inference_response(
            inference_service=http_s3_tensorflow_model_mesh_inference_service,
            runtime=ModelInferenceRuntime.TENSORFLOW_RUNTIME,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )
