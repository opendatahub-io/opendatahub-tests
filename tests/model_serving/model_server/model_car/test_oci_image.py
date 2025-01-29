import pytest

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.infra import get_pods_by_isvc_label
from utilities.constants import ModelName, Protocols, ModelInferenceRuntime
from utilities.inference_utils import Inference


pytestmark = pytest.mark.serverless


@pytest.mark.parametrize(
    "model_namespace, serving_runtime_from_template, model_car_tgis_inference_service",
    [
        pytest.param(
            {"name": "tgsi-model-car"},
            {
                "name": "tgis-runtime",
                "template-name": "tgis-grpc-serving-template",
                "multi-model": False,
            },
            {
                "storage-uri": "oci://quay.io/mwaykole/test@sha256:c526a1a3697253eb09adc65da6efaf7f36150205c3a51ab8d13b92b6a3af9c1c"  # noqa: E501
            },
        )
    ],
    indirect=True,
)
class TestKserveModelCar:
    @pytest.mark.smoke
    @pytest.mark.jira("RHOAIENG-13465")
    def test_model_car_no_restarts(self, model_car_tgis_inference_service):
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
            runtime=ModelInferenceRuntime.TGIS_RUNTIME,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.GRPC,
            model_name=ModelName.FLAN_T5_SMALL_HF,
            use_default_query=True,
        )
