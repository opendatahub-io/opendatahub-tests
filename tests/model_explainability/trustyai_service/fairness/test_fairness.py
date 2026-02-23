from functools import partial
from typing import Any

import pytest
from ocp_resources.inference_service import InferenceService

from tests.model_explainability.trustyai_service.trustyai_service_utils import (
    TrustyAIServiceMetrics,
    send_inferences_and_verify_trustyai_service_registered,
    verify_trustyai_service_metric_delete_request,
    verify_trustyai_service_metric_request,
    verify_trustyai_service_metric_scheduling_request,
    verify_trustyai_service_name_mappings,
)
from utilities.constants import MinIo
from utilities.manifests.openvino import OPENVINO_KSERVE_INFERENCE_CONFIG
from utilities.monitoring import get_metric_label, validate_metrics_field

BASE_DATA_PATH: str = "./tests/model_explainability/trustyai_service/fairness/model_data"
IS_MALE_IDENTIFYING: str = "Is Male-Identifying?"
WILL_DEFAULT: str = "Will Default?"
INPUT_NAME_MAPPINGS: dict[str, str] = {
    "customer_data_input-0": "Number of Children",
    "customer_data_input-1": "Total Income",
    "customer_data_input-2": "Number of Total Family Members",
    "customer_data_input-3": IS_MALE_IDENTIFYING,
    "customer_data_input-4": "Owns Car?",
    "customer_data_input-5": "Owns Realty?",
    "customer_data_input-6": "Is Partnered?",
    "customer_data_input-7": "Is Employed?",
    "customer_data_input-8": "Live with Parents?",
    "customer_data_input-9": "Age",
    "customer_data_input-10": "Length of Employment?",
}
OUTPUT_NAME_MAPPINGS: dict[str, str] = {"predict": WILL_DEFAULT}

FAIRNESS_METRICS = [TrustyAIServiceMetrics.Fairness.SPD, TrustyAIServiceMetrics.Fairness.DIR]


def get_fairness_request_json_data(isvc: InferenceService) -> dict[str, Any]:
    return {
        "modelId": isvc.name,
        "protectedAttribute": IS_MALE_IDENTIFYING,
        "privilegedAttribute": 1.0,
        "unprivilegedAttribute": 0.0,
        "outcomeName": WILL_DEFAULT,
        "favorableOutcome": 0,
        "batchSize": 5000,
    }


@pytest.mark.usefixtures("minio_pod")
@pytest.mark.parametrize(
    "model_namespace, minio_pod, minio_data_connection, trustyai_service",
    [
        pytest.param(
            {"name": "test-fairness-pvc"},
            MinIo.PodConfig.MODEL_MESH_MINIO_CONFIG,
            {"bucket": MinIo.Buckets.MODELMESH_EXAMPLE_MODELS},
            {"storage": "pvc"},
            id="pvc-storage",
        ),
        pytest.param(
            {"name": "test-fairness-db"},
            MinIo.PodConfig.MODEL_MESH_MINIO_CONFIG,
            {"bucket": MinIo.Buckets.MODELMESH_EXAMPLE_MODELS},
            {"storage": "db"},
            id="db-storage",
        ),
    ],
    indirect=True,
)
@pytest.mark.usefixtures("minio_pod")
@pytest.mark.rawdeployment
@pytest.mark.smoke
class TestFairnessMetrics:
    """
    Verifies all the basic operations that can be performed with all the fairness metrics
    (SPD and DIR) available in TrustyAI.

    1. Send data to the model and verify that TrustyAI registers the observations.
    2. Apply name mappings
    3. Send metric request (SPD and DIR) and verify the response.
    4. Send metric scheduling request and verify the response.
    5. Send metric deletion request and verify that the scheduled metric has been deleted.

    The tests are run for both PVC and DB storage.
    """

    @pytest.mark.dependency(name="send_inference")
    def test_fairness_send_inference(
        self,
        admin_client,
        current_client_token,
        model_namespace,
        trustyai_service,
        onnx_loan_model,
        isvc_getter_token,
    ):
        send_inferences_and_verify_trustyai_service_registered(
            client=admin_client,
            token=current_client_token,
            data_path=f"{BASE_DATA_PATH}",
            trustyai_service=trustyai_service,
            inference_service=onnx_loan_model,
            inference_config=OPENVINO_KSERVE_INFERENCE_CONFIG,
            inference_token=isvc_getter_token,
        )

    @pytest.mark.dependency(name="upload_data", depends=["send_inference"])
    def test_name_mappings(
        self, admin_client, current_client_token, model_namespace, trustyai_service, onnx_loan_model
    ):
        verify_trustyai_service_name_mappings(
            client=admin_client,
            token=current_client_token,
            trustyai_service=trustyai_service,
            isvc=onnx_loan_model,
            input_mappings=INPUT_NAME_MAPPINGS,
            output_mappings=OUTPUT_NAME_MAPPINGS,
        )

    @pytest.mark.dependency(depends=["upload_data"])
    @pytest.mark.parametrize("metric_name", FAIRNESS_METRICS)
    def test_fairness_metric(self, admin_client, current_client_token, trustyai_service, onnx_loan_model, metric_name):
        verify_trustyai_service_metric_request(
            client=admin_client,
            trustyai_service=trustyai_service,
            token=current_client_token,
            metric_name=metric_name,
            json_data=get_fairness_request_json_data(isvc=onnx_loan_model),
        )

    @pytest.mark.dependency(name="schedule_metric", depends=["upload_data"])
    @pytest.mark.parametrize("metric_name", FAIRNESS_METRICS)
    def test_fairness_metric_schedule(
        self, admin_client, current_client_token, trustyai_service, onnx_loan_model, metric_name
    ):
        verify_trustyai_service_metric_scheduling_request(
            client=admin_client,
            trustyai_service=trustyai_service,
            token=current_client_token,
            metric_name=metric_name,
            json_data=get_fairness_request_json_data(isvc=onnx_loan_model),
        )

    @pytest.mark.dependency(depends=["schedule_metric"])
    @pytest.mark.parametrize("metric_name", FAIRNESS_METRICS)
    def test_fairness_metric_prometheus(
        self, admin_client, model_namespace, trustyai_service, onnx_loan_model, prometheus, metric_name
    ):
        validate_metrics_field(
            prometheus=prometheus,
            metrics_query=f'trustyai_{metric_name}{{namespace="{model_namespace.name}"}}',
            expected_value=metric_name.upper(),
            field_getter=partial(get_metric_label, label_name="metricName"),
        )

    @pytest.mark.dependency(depends=["schedule_metric"])
    @pytest.mark.parametrize("metric_name", FAIRNESS_METRICS)
    def test_fairness_metric_delete(
        self, admin_client, current_client_token, trustyai_service, onnx_loan_model, metric_name
    ):
        verify_trustyai_service_metric_delete_request(
            client=admin_client,
            trustyai_service=trustyai_service,
            token=current_client_token,
            metric_name=metric_name,
        )
