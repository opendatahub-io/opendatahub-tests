import pytest
from _pytest.fixtures import FixtureRequest

from tests.model_serving.model_server.kserve.storage.utils import assert_isvc_oci_injected
from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import KServeDeploymentType, ModelCarImage, ModelFormat, ModelName, Protocols, RuntimeTemplates
from utilities.inference_utils import Inference
from utilities.infra import get_pods_by_isvc_label
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG


@pytest.mark.parametrize(
    "unprivileged_model_namespace, serving_runtime_from_template, model_car_inference_service",
    [
        pytest.param(
            {"name": f"{ModelFormat.OPENVINO}-model-car"},
            {
                "name": f"{ModelName.MNIST}-runtime",
                "template-name": RuntimeTemplates.OVMS_KSERVE,
                "multi-model": False,
            },
            {
                # Using mnist-8-1 model from OCI image
                "storage-uri": ModelCarImage.MNIST_8_1,
                "deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT,
            },
            marks=[pytest.mark.rawdeployment],
            id="rawdeployment",
        ),
        pytest.param(
            {"name": f"{ModelFormat.OPENVINO}-model-car-connection"},
            {
                "name": f"{ModelName.MNIST}-runtime",
                "template-name": RuntimeTemplates.OVMS_KSERVE,
                "multi-model": False,
            },
            {
                # OCI injection only adds imagePullSecrets — storage-uri stays the real modelcar
                # reference (manual 1.3).
                "storage-uri": ModelCarImage.MNIST_8_1,
                "deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT,
                "connection-secret-fixture": "oci_connection_secret",
            },
            id="connection",
        ),
    ],
    indirect=True,
)
class TestKserveModelCar:
    """Validate KServe model serving using OCI Model Car images for model storage.

    Steps:
        1. Deploy an OVMS inference service using an OCI Model Car image (MNIST), either with a
           static `storage-uri` or (for the `connection` param) a ConnectionsAPI connection Secret.
        2. Verify the predictor pod does not experience excessive container restarts.
        3. For the `connection` param, assert the ConnectionsAPI webhook injected
           `imagePullSecrets`, then send a REST inference request and verify a successful response.
        4. Verify the model status on the InferenceService resource is Loaded and UpToDate.
    """

    @pytest.mark.tier1
    def test_model_car_no_restarts(self, model_car_inference_service):
        """Verify that model pod doesn't restart"""
        pod = get_pods_by_isvc_label(
            client=model_car_inference_service.client,
            isvc=model_car_inference_service,
        )[0]
        restarted_containers = [
            container.name for container in pod.instance.status.containerStatuses if container.restartCount > 2
        ]
        assert not restarted_containers, f"Containers {restarted_containers} restarted"

    @pytest.mark.tier1
    @pytest.mark.ocp_interop
    def test_model_car_using_rest(self, request: FixtureRequest, model_car_inference_service):
        """Verify model query with token using REST"""
        isvc_param = request.node.callspec.params["model_car_inference_service"]
        if connection_secret_fixture := isvc_param.get("connection-secret-fixture"):
            secret = request.getfixturevalue(argname=connection_secret_fixture)
            assert_isvc_oci_injected(isvc=model_car_inference_service, secret_name=secret.name)

        verify_inference_response(
            inference_service=model_car_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )

    @pytest.mark.tier1
    @pytest.mark.ocp_interop
    def test_model_status_loaded(self, model_car_inference_service):
        """Verify model status on the InferenceService resource is in a valid state."""
        model_status = model_car_inference_service.instance.status.modelStatus

        # After deployment the model should be fully loaded and up to date.
        assert model_status.states.activeModelState == "Loaded"
        assert model_status.states.targetModelState == "Loaded"
        assert model_status.transitionStatus == "UpToDate"
