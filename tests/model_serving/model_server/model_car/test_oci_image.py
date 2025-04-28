import pytest

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.infra import get_pods_by_isvc_label
from utilities.constants import ModelFormat, ModelName, Protocols, RuntimeTemplates
from utilities.inference_utils import Inference
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG

pytestmark = pytest.mark.serverless


@pytest.mark.parametrize(
    "unprivileged_model_namespace, serving_runtime_from_template, model_car_serverless_inference_service",
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
                "storage-uri": "oci://quay.io/mwaykole/test@sha256:8a3217bcfa2cc5fa3d07496cff8b234acdf2c9725dd307dc0a80401f55e1a11c"  # noqa: E501
            },
        )
    ],
    indirect=True,
)
class TestKserveModelCar:
    @pytest.mark.smoke
    @pytest.mark.jira("RHOAIENG-13465")
    def test_model_car_no_restarts(self, model_car_serverless_inference_service):
        """Verify that model pod doesn't restart"""
        pod = get_pods_by_isvc_label(
            client=model_car_serverless_inference_service.client,
            isvc=model_car_serverless_inference_service,
        )[0]
        restarted_containers = [
            container.name for container in pod.instance.status.containerStatuses if container.restartCount > 1
        ]
        assert not restarted_containers, f"Containers {restarted_containers} restarted"

    @pytest.mark.smoke
    @pytest.mark.ocp_interop
    @pytest.mark.jira("RHOAIENG-12306")
    def test_model_car_using_rest(self, model_car_serverless_inference_service):
        """Verify model query with token using REST"""
        verify_inference_response(
            inference_service=model_car_serverless_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )
