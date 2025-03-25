import pytest

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.infra import get_pods_by_isvc_label
from utilities.constants import ModelInferenceRuntime, ModelName, Protocols
from utilities.inference_utils import Inference
from utilities.manifests.tgis_grpc import TGIS_INFERENCE_CONFIG

pytestmark = pytest.mark.serverless


@pytest.mark.parametrize(
    "model_namespace, serving_runtime_from_template, model_car_tgis_inference_service",
    [
        pytest.param(
            {"name": "openvino-model-car"},
            {
                "name": "tgis-runtime",
                "template-name": ModelInferenceRuntime.ONNX_RUNTIME,
                "multi-model": False,
            },
            {
                "storage-uri": "oci://quay.io/mwaykole/test:mnist-8-1"  # noqa: E501
            },
        )
    ],
    indirect=True,
)
class TestKserveModelCar:
    @pytest.mark.smoke
    @pytest.mark.jira("RHOAIENG-13465")
    def test_model_car_no_restarts(self, model_car_tgis_inference_service):
        """Verify that model pod doesn't restart"""
        pod = get_pods_by_isvc_label(
            client=model_car_tgis_inference_service.client,
            isvc=model_car_tgis_inference_service,
        )[0]
        restarted_containers = [
            container.name for container in pod.instance.status.containerStatuses if container.restartCount > 1
        ]
        assert not restarted_containers, f"Containers {restarted_containers} restarted"

    @pytest.mark.smoke
    @pytest.mark.jira("RHOAIENG-12306")
    def test_model_car_using_rest(self, model_car_tgis_inference_service):
        """Verify model query with token using REST"""
        verify_inference_response(
            inference_service=model_car_tgis_inference_service,
            inference_config=TGIS_INFERENCE_CONFIG,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.GRPC,
            model_name=ModelName.FLAN_T5_SMALL_HF,
            use_default_query=True,
        )
