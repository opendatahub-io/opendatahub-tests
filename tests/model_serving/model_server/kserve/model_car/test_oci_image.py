import pytest

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
        ),
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
        ),
    ],
    indirect=True,
)
class TestKserveModelCar:
    @pytest.mark.smoke
    @pytest.mark.jira("RHOAIENG-13465")
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

    @pytest.mark.smoke
    @pytest.mark.ocp_interop
    @pytest.mark.jira("RHOAIENG-12306")
    def test_model_car_using_rest(self, model_car_inference_service):
        """Verify model query with token using REST"""
        verify_inference_response(
            inference_service=model_car_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )

    @pytest.mark.smoke
    @pytest.mark.ocp_interop
    @pytest.mark.jira("RHOAIENG-38674")
    def test_model_status_loaded(self, model_car_inference_service):
        """Verify model status on the InferenceService resource is in a valid state."""
        model_status = model_car_inference_service.instance.status.modelStatus

        # After deployment the model should be fully loaded and up to date.
        assert model_status.states.activeModelState == "Loaded"
        assert model_status.states.targetModelState == "Loaded"
        assert model_status.transitionStatus == "UpToDate"
