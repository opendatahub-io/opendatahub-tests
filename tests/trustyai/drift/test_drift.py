import pytest

from tests.trustyai.drift.utils import (
    send_inference_requests_and_verify_trustyai_service,
    verify_trustyai_metric_scheduling_request,
    verify_trustyai_metric_request,
    verify_upload_data_to_trustyai_service,
    verify_trustyai_drift_metric_delete_request,
)

MEANSHIFT: str = "meanshift"
BASE_DATA_PATH: str = "./tests/trustyai/drift/model_data"


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-drift-gaussian-credit-model", "modelmesh-enabled": True},
        )
    ],
    indirect=True,
)
class TestDriftMetrics:
    """
    Verifies all the basic operations with a drift metric (meanshift) available in TrustyAI, using PVC storage.

    1. Send data to the model (gaussian_credit_model) and verify that TrustyAI registers the observations.
    2. Send metric request (meanshift) and verify the response.
    3. Send metric scheduling request and verify the response.
    4. Send metric deletion request and verify that the scheduled metric has been deleted.
    """

    def test_send_inference_request_and_verify_trustyai_service(
        self,
        admin_client,
        current_client_token,
        model_namespace,
        trustyai_service_with_pvc_storage,
        gaussian_credit_model,
    ) -> None:
        send_inference_requests_and_verify_trustyai_service(
            client=admin_client,
            token=current_client_token,
            data_path=f"{BASE_DATA_PATH}/data_batches",
            trustyai_service=trustyai_service_with_pvc_storage,
            inference_service=gaussian_credit_model,
        )

    def test_upload_data_to_trustyai_service(
        self,
        admin_client,
        current_client_token,
        trustyai_service_with_pvc_storage,
    ) -> None:
        verify_upload_data_to_trustyai_service(
            client=admin_client,
            trustyai_service=trustyai_service_with_pvc_storage,
            token=current_client_token,
            data_path=f"{BASE_DATA_PATH}/training_data.json",
        )

    def test_drift_metric_meanshift(
        self, admin_client, current_client_token, trustyai_service_with_pvc_storage, gaussian_credit_model
    ):
        verify_trustyai_metric_request(
            client=admin_client,
            trustyai_service=trustyai_service_with_pvc_storage,
            token=current_client_token,
            metric_name=MEANSHIFT,
            json_data={"modelId": gaussian_credit_model.name, "referenceTag": "TRAINING"},
        )

    def test_drift_metric_schedule_meanshift(
        self, admin_client, current_client_token, trustyai_service_with_pvc_storage, gaussian_credit_model
    ):
        verify_trustyai_metric_scheduling_request(
            client=admin_client,
            trustyai_service=trustyai_service_with_pvc_storage,
            token=current_client_token,
            metric_name=MEANSHIFT,
            json_data={"modelId": gaussian_credit_model.name, "referenceTag": "TRAINING"},
        )

    def test_drift_metric_delete(self, admin_client, current_client_token, trustyai_service_with_pvc_storage):
        verify_trustyai_drift_metric_delete_request(
            client=admin_client,
            trustyai_service=trustyai_service_with_pvc_storage,
            token=current_client_token,
            metric_name=MEANSHIFT,
        )
