import pytest

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import (
    ModelStoragePath,
    Protocols,
)
from utilities.inference_utils import Inference
from utilities.manifests.openvino import OPENVINO_INFERENCE_CONFIG
from utilities.manifests.tensorflow import TENSORFLOW_INFERENCE_CONFIG

pytestmark = [pytest.mark.modelmesh]


@pytest.mark.parametrize(
    "model_namespace, http_s3_ovms_model_mesh_serving_runtime, http_s3_openvino_model_mesh_inference_service, "
    "http_s3_tensorflow_model_mesh_inference_service",
    [
        pytest.param(
            {"name": "model-mesh-openvino", "modelmesh-enabled": True},
            {"enable-external-route": True},
            {"model-path": ModelStoragePath.OPENVINO_EXAMPLE_MODEL},
            {
                "model-path": ModelStoragePath.TENSORFLOW_MODEL,
                "runtime-fixture-name": "http_s3_ovms_external_route_model_mesh_serving_runtime",
            },
        )
    ],
    indirect=True,
)
class TestOpenVINOModelMeshMultiServers:
    def test_model_mesh_openvino_rest_multi_servers(
        self,
        http_s3_openvino_model_mesh_inference_service,
        http_s3_tensorflow_model_mesh_inference_service,
    ):
        verify_inference_response(
            inference_service=http_s3_openvino_model_mesh_inference_service,
            inference_config=OPENVINO_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTP,
            use_default_query=True,
        )

    def test_model_mesh_tensorflow_rest_multi_servers(
        self,
        http_s3_tensorflow_model_mesh_inference_service,
        http_s3_openvino_model_mesh_inference_service,
    ):
        verify_inference_response(
            inference_service=http_s3_tensorflow_model_mesh_inference_service,
            inference_config=TENSORFLOW_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )
