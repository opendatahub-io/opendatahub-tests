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
    "model_namespace, http_s3_openvino_model_mesh_inference_service, "
    "http_s3_openvino_second_model_mesh_inference_service",
    [
        pytest.param(
            {"name": "model-mesh-multi-model", "modelmesh-enabled": True},
            {"model-path": ModelStoragePath.OPENVINO_EXAMPLE_MODEL},
            {"model-path": ModelStoragePath.OPENVINO_VEHICLE_DETECTION},
        )
    ],
    indirect=True,
)
class TestOpenVINOModelMeshMultiModels:
    def test_model_mesh_openvino_inference_with_tensorflow(
        self,
        http_s3_openvino_model_mesh_inference_service,
        http_s3_openvino_second_model_mesh_inference_service,
    ):
        verify_inference_response(
            inference_service=http_s3_openvino_model_mesh_inference_service,
            runtime=ModelInferenceRuntime.OPENVINO_RUNTIME,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTP,
            use_default_query=True,
        )

    def test_model_mesh_tensorflow_with_openvino(
        self,
        http_s3_openvino_second_model_mesh_inference_service,
        http_s3_openvino_model_mesh_inference_service,
    ):
        verify_inference_response(
            inference_service=http_s3_openvino_second_model_mesh_inference_service,
            runtime=ModelInferenceRuntime.TENSORFLOW_RUNTIME,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTP,
            query_input="@utilities/manifests/openvino/vehicle-detection-inputs.txt",
        )
