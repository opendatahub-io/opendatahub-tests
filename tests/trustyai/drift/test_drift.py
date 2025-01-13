import pytest

from tests.trustyai.drift.utils import send_inference_requests_and_verify_trustyai_service, verify_metric_request


@pytest.mark.parametrize(
    "ns_with_modelmesh_enabled",
    [
        pytest.param(
            {"name": "test-drift-gaussian-credit-model"},
        )
    ],
    indirect=True,
)
class TestDriftMetrics:
    """
    Verifies all the basic operations with a drift metric (meanshift) available in TrustyAI, using PVC storage.

    1. Send data to the model (gaussian_credit_model) and verify that TrustyAI registers the observations.
    2. Send metric request (meanshift) and verify the response.
    """

    def test_send_inference_request_and_verify_trustyai_service(
        self,
        admin_client,
        current_client_token,
        ns_with_modelmesh_enabled,
        trustyai_service_with_pvc_storage,
        gaussian_credit_model,
    ) -> None:
        send_inference_requests_and_verify_trustyai_service(
            client=admin_client,
            token=current_client_token,
            data_path="./tests/trustyai/drift/model_data/data_batches",
            trustyai_service=trustyai_service_with_pvc_storage,
            inference_service=gaussian_credit_model,
        )

        # TODO: Add rest of operations in upcoming PRs
        #  (upload data directly to Trusty, send metric request,
        #  schedule period metric calculation, delete metric request).

    def test_drift_metric_meanshift(
        self, admin_client, current_client_token, trustyai_service_with_pvc_storage, gaussian_credit_model
    ):
        verify_metric_request(
            client=admin_client,
            trustyai_service=trustyai_service_with_pvc_storage,
            token=current_client_token,
            metric_name="meanshift",
            json_data={"modelId": gaussian_credit_model.name, "referenceTag": "TRAINING"},
        )
