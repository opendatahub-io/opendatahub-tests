import pytest

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.inference_utils import Inference
from utilities.constants import ModelInferenceRuntime, Protocols
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG


@pytest.mark.serverless
@pytest.mark.sanity
@pytest.mark.parametrize(
    "unprivileged_model_namespace,ovms_kserve_serving_runtime",
    [
        pytest.param(
            {"name": "kserve-inference-graph-deploy"},
            {"runtime-name": ModelInferenceRuntime.ONNX_RUNTIME},
        )
    ],
    indirect=True,
)
class TestInferenceGraphDeployment:
    @pytest.mark.parametrize(
        "dog_breed_inference_graph",
        [{"name": "dog-breed-serverless-pipeline"}],
        indirect=True,
    )
    def test_inference_graph_serverless_deployment(self, dog_breed_inference_graph):
        verify_inference_response(
            inference_service=dog_breed_inference_graph,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.GRAPH,
            model_name="dog-breed-classifier",
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )

    @pytest.mark.parametrize(
        "dog_breed_inference_graph",
        [{"name": "dog-breed-private-serverless-ig", "external-route": False}],
        indirect=True,
    )
    def test_private_inference_graph_serverless_deployment(self, dog_breed_inference_graph):
        verify_inference_response(
            inference_service=dog_breed_inference_graph,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.GRAPH,
            model_name="dog-breed-classifier",
            protocol=Protocols.HTTP,
            use_default_query=True,
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "dog_breed_inference_graph",
        [{"name": "dog-breed-auth-serverless-ig", "enable-auth": True}],
        indirect=True,
    )
    def test_inference_graph_serverless_authentication(
        self, dog_breed_inference_graph, inference_graph_sa_token_with_access
    ):
        verify_inference_response(
            inference_service=dog_breed_inference_graph,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.GRAPH,
            model_name="dog-breed-classifier",
            protocol=Protocols.HTTPS,
            use_default_query=True,
            token=inference_graph_sa_token_with_access,
        )

    @pytest.mark.parametrize(
        "dog_breed_inference_graph",
        [{"name": "dog-breed-bad-auth-serverless-ig", "enable-auth": True}],
        indirect=True,
    )
    def test_inference_graph_serverless_authentication_without_privileges(
        self, dog_breed_inference_graph, inference_graph_unprivileged_sa_token
    ):
        verify_inference_response(
            inference_service=dog_breed_inference_graph,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.GRAPH,
            model_name="dog-breed-classifier",
            protocol=Protocols.HTTPS,
            use_default_query=True,
            token=inference_graph_unprivileged_sa_token,
            authorized_user=False,
        )
