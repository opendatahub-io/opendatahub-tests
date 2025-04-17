import pytest

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.inference_utils import Inference
from utilities.constants import ModelInferenceRuntime, Protocols
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG


@pytest.mark.parametrize(
    "model_namespace,ovms_kserve_serving_runtime",
    [pytest.param({"name": "kserve-inference-graph-deploy"}, {"runtime-name": ModelInferenceRuntime.ONNX_RUNTIME})],
    indirect=True,
)
class TestInferenceGraphDeployment:
    def test_inference_graph_deployment(self, dog_breed_inference_graph):
        verify_inference_response(
            inference_service=dog_breed_inference_graph,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.GRAPH,
            model_name="dog-breed-classifier",
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )
