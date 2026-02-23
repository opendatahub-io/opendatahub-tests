import pytest

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import KServeDeploymentType, ModelInferenceRuntime, Protocols
from utilities.inference_utils import Inference
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG


@pytest.mark.rawdeployment
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
class TestInferenceGraphRaw:
    @pytest.mark.parametrize(
        "dog_breed_inference_graph",
        [{"name": "dog-breed-raw-pipeline", "deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT}],
        indirect=True,
    )
    def test_inference_graph_raw_deployment(self, dog_breed_inference_graph):
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
        [
            {
                "name": "dog-breed-private-raw-pipeline",
                "deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT,
                "external-route": False,
            }
        ],
        indirect=True,
    )
    def test_private_inference_graph_raw_deployment(self, dog_breed_inference_graph):
        verify_inference_response(
            inference_service=dog_breed_inference_graph,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.GRAPH,
            model_name="dog-breed-classifier",
            protocol=Protocols.HTTPS,
            use_default_query=True,
            insecure=True,
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "dog_breed_inference_graph",
        [
            {
                "name": "dog-breed-auth-raw-ig",
                "deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT,
                "enable-auth": True,
            }
        ],
        indirect=True,
    )
    def test_inference_graph_raw_authentication(self, dog_breed_inference_graph, inference_graph_sa_token_with_access):
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
        [
            {
                "name": "dog-breed-private-auth-raw-ig",
                "deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT,
                "enable-auth": True,
                "external-route": False,
            }
        ],
        indirect=True,
    )
    def test_private_inference_graph_raw_authentication(
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
            insecure=True,
        )

    @pytest.mark.parametrize(
        "dog_breed_inference_graph",
        [
            {
                "name": "dog-breed-bad-auth-raw-ig",
                "deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT,
                "enable-auth": True,
            }
        ],
        indirect=True,
    )
    def test_inference_graph_raw_authentication_without_privileges(
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

    @pytest.mark.parametrize(
        "dog_breed_inference_graph",
        [
            {
                "name": "dog-breed-private-bad-auth-raw-ig",
                "deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT,
                "enable-auth": True,
                "external-route": False,
            }
        ],
        indirect=True,
    )
    def test_private_inference_graph_raw_authentication_without_privileges(
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
            insecure=True,
        )
