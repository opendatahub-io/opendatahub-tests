import pytest

from tests.trustyai.drift.utils import send_inference_requests_and_verify_trustyai_service


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
    """

    def test_send_inference_request_and_verify_trustyai_service(
        self,
        admin_client,
        openshift_token,
        ns_with_modelmesh_enabled,
        trustyai_service_with_pvc_storage,
        gaussian_credit_model,
    ) -> None:
        send_inference_requests_and_verify_trustyai_service(
            client=admin_client,
            token=openshift_token,
            data_path="./tests/trustyai/drift/model_data/data_batches",
            trustyai_service=trustyai_service_with_pvc_storage,
            inference_service=gaussian_credit_model,
        )

        # TODO: Add rest of operations in upcoming PRs (upload data directly to Trusty, send metric request, schedule period metric calculation, delete metric request).
