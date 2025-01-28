import pytest

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import (
    ModelStoragePath,
    Protocols,
)
from utilities.inference_utils import Inference
from utilities.manifests.tensorflow import TENSORFLOW_INFERENCE_CONFIG

pytestmark = [pytest.mark.modelmesh]


@pytest.mark.parametrize(
    "model_namespace, http_s3_ovms_model_mesh_serving_runtime, http_s3_tensorflow_model_mesh_inference_service",
    [
        pytest.param(
            {"name": "model-mesh-tensorflow", "modelmesh-enabled": True},
            {"enable-external-route": True},
            {"model-path": ModelStoragePath.TENSORFLOW_MODEL},
        )
    ],
    indirect=True,
)
class TestTensorflowModelMesh:
    @pytest.mark.sanity
    @pytest.mark.polarion("ODS-2268")
    def test_model_mesh_tensorflow_rest_inference_external_route(self, http_s3_tensorflow_model_mesh_inference_service):
        verify_inference_response(
            inference_service=http_s3_tensorflow_model_mesh_inference_service,
            inference_config=TENSORFLOW_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )
