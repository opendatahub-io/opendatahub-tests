from typing import Any

import pytest
from ocp_resources.inference_service import InferenceService

from tests.model_explainability.trustyai_service.trustyai_service_utils import (
    send_inferences_and_verify_trustyai_service_registered,
    verify_trustyai_service_name_mappings,
    verify_trustyai_service_metric_request,
    TrustyAIServiceMetrics,
    verify_trustyai_service_metric_delete_request,
    verify_trustyai_service_metric_scheduling_request,
)
from utilities.constants import MinIo
from utilities.manifests.openvino import OPENVINO_KSERVE_INFERENCE_CONFIG

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
    "model_namespace, minio_pod, minio_data_connection",
    [
        pytest.param(
            {"name": "test-fairness-pvc"},
            MinIo.PodConfig.MODEL_MESH_MINIO_CONFIG,
            {"bucket": MinIo.Buckets.MODELMESH_EXAMPLE_MODELS},
        )
    ],
    indirect=True,
)
@pytest.mark.usefixtures("minio_pod")
class TestFairnessMetricsWithPVCStorage:
    """
    Verifies all the basic operations with a fairness metric (spd) available in TrustyAI, using PVC storage.

    1. Send data to the model and verify that TrustyAI registers the observations.
    2. Apply name mappings
    3. Send metric request (spd) and verify the response.
    4. Send metric scheduling request and verify the response.
    5. Send metric deletion request and verify that the scheduled metric has been deleted.
    """

    def test_fairness_send_inference_and_verify_trustyai_service_with_pvc_storage(
        self, admin_client, current_client_token, model_namespace, trustyai_service_with_pvc_storage, onnx_loan_model
    ):
        send_inferences_and_verify_trustyai_service_registered(
            client=admin_client,
            token=current_client_token,
            data_path=f"{BASE_DATA_PATH}",
            trustyai_service=trustyai_service_with_pvc_storage,
            inference_service=onnx_loan_model,
            inference_config=OPENVINO_KSERVE_INFERENCE_CONFIG,
        )

    def test_name_mappings_with_pvc_storage(
        self, admin_client, current_client_token, model_namespace, trustyai_service_with_pvc_storage, onnx_loan_model
    ):
        verify_trustyai_service_name_mappings(
            client=admin_client,
            token=current_client_token,
            trustyai_service=trustyai_service_with_pvc_storage,
            isvc=onnx_loan_model,
            input_mappings=INPUT_NAME_MAPPINGS,
            output_mappings=OUTPUT_NAME_MAPPINGS,
        )

    def test_fairness_metric_spd_with_pvc_storage(
        self, admin_client, current_client_token, trustyai_service_with_pvc_storage, onnx_loan_model
    ):
        verify_trustyai_service_metric_request(
            client=admin_client,
            trustyai_service=trustyai_service_with_pvc_storage,
            token=current_client_token,
            metric_name=TrustyAIServiceMetrics.Fairness.SPD,
            json_data=get_fairness_request_json_data(isvc=onnx_loan_model),
        )

    def test_fairness_metric_schedule_spd_with_pvc_storage(
        self, admin_client, current_client_token, trustyai_service_with_pvc_storage, onnx_loan_model
    ):
        verify_trustyai_service_metric_scheduling_request(
            client=admin_client,
            trustyai_service=trustyai_service_with_pvc_storage,
            token=current_client_token,
            metric_name=TrustyAIServiceMetrics.Fairness.SPD,
            json_data=get_fairness_request_json_data(isvc=onnx_loan_model),
        )

    def test_fairness_metric_delete_with_pvc_storage(
        self, admin_client, current_client_token, trustyai_service_with_pvc_storage, onnx_loan_model
    ):
        verify_trustyai_service_metric_delete_request(
            client=admin_client,
            trustyai_service=trustyai_service_with_pvc_storage,
            token=current_client_token,
            metric_name=TrustyAIServiceMetrics.Fairness.SPD,
        )


@pytest.mark.parametrize(
    "model_namespace, minio_pod, minio_data_connection",
    [
        pytest.param(
            {"name": "test-fairness-db"},
            MinIo.PodConfig.MODEL_MESH_MINIO_CONFIG,
            {"bucket": MinIo.Buckets.MODELMESH_EXAMPLE_MODELS},
        )
    ],
    indirect=True,
)
@pytest.mark.usefixtures("minio_pod")
class TestFairnessMetricsWithDBStorage:
    """
    Verifies all the basic operations with a fairness metric (spd) available in TrustyAI, using MariaDB storage.

    1. Send data to the model and verify that TrustyAI registers the observations.
    2. Apply name mappings
    3. Send metric request (spd) and verify the response.
    4. Send metric scheduling request and verify the response.
    5. Send metric deletion request and verify that the scheduled metric has been deleted.
    """

    def test_fairness_send_inference_and_verify_trustyai_service_with_db_storage(
        self, admin_client, current_client_token, model_namespace, trustyai_service_with_db_storage, onnx_loan_model
    ):
        send_inferences_and_verify_trustyai_service_registered(
            client=admin_client,
            token=current_client_token,
            data_path=f"{BASE_DATA_PATH}",
            trustyai_service=trustyai_service_with_db_storage,
            inference_service=onnx_loan_model,
            inference_config=OPENVINO_KSERVE_INFERENCE_CONFIG,
        )

    def test_name_mappings_with_db_storage(
        self, admin_client, current_client_token, model_namespace, trustyai_service_with_db_storage, onnx_loan_model
    ):
        verify_trustyai_service_name_mappings(
            client=admin_client,
            token=current_client_token,
            trustyai_service=trustyai_service_with_db_storage,
            isvc=onnx_loan_model,
            input_mappings=INPUT_NAME_MAPPINGS,
            output_mappings=OUTPUT_NAME_MAPPINGS,
        )

    def test_fairness_metric_spd_with_db_storage(
        self, admin_client, current_client_token, trustyai_service_with_db_storage, onnx_loan_model
    ):
        verify_trustyai_service_metric_request(
            client=admin_client,
            trustyai_service=trustyai_service_with_db_storage,
            token=current_client_token,
            metric_name=TrustyAIServiceMetrics.Fairness.SPD,
            json_data=get_fairness_request_json_data(isvc=onnx_loan_model),
        )

    def test_fairness_metric_schedule_spd_with_db_storage(
        self, admin_client, current_client_token, trustyai_service_with_db_storage, onnx_loan_model
    ):
        verify_trustyai_service_metric_scheduling_request(
            client=admin_client,
            trustyai_service=trustyai_service_with_db_storage,
            token=current_client_token,
            metric_name=TrustyAIServiceMetrics.Fairness.SPD,
            json_data=get_fairness_request_json_data(isvc=onnx_loan_model),
        )

    def test_fairness_metric_delete_with_db_storage(
        self, admin_client, current_client_token, trustyai_service_with_db_storage, onnx_loan_model
    ):
        verify_trustyai_service_metric_delete_request(
            client=admin_client,
            trustyai_service=trustyai_service_with_db_storage,
            token=current_client_token,
            metric_name=TrustyAIServiceMetrics.Fairness.SPD,
        )
