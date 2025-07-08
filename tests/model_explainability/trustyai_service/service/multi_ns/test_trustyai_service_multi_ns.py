import pytest
from tests.model_explainability.trustyai_service.constants import (
    DRIFT_BASE_DATA_PATH
)
from tests.model_explainability.trustyai_service.trustyai_service_utils import (
    send_inferences_and_verify_trustyai_service_registered,
    verify_upload_data_to_trustyai_service,
    TrustyAIServiceMetrics,
    verify_trustyai_service_metric_scheduling_request,
    verify_trustyai_service_metric_delete_request,
)
from utilities.constants import MinIo
from utilities.manifests.openvino import OPENVINO_KSERVE_INFERENCE_CONFIG

@pytest.mark.usefixtures("minio_pod")
@pytest.mark.parametrize(
    "model_namespaces, minio_pod, minio_data_connection_multi_ns",
    [
        pytest.param(
            [
                {"name": "test-trustyaiservice-multins-1"},
                {"name": "test-trustyaiservice-multins-2"},
            ],
            MinIo.PodConfig.MODEL_MESH_MINIO_CONFIG,
            [
                {"bucket": MinIo.Buckets.MODELMESH_EXAMPLE_MODELS},
                {"bucket": MinIo.Buckets.MODELMESH_EXAMPLE_MODELS},
            ]
        ),
    ],
    indirect=True,
)
class TestTrustyAIServiceMultipleNS:
    """Verifies TrustyAIService operations across multiple namespaces."""

    @pytest.mark.parametrize("ns_index", [0, 1])
    def test_drift_send_inference_and_verify_trustyai_service_multiple_ns(
        self,
        ns_index: int,
        admin_client,
        current_client_token,
        trustyai_service_with_pvc_storage_multi_ns,
        gaussian_credit_model_multi_ns,
        isvc_getter_token_multi_ns,
        model_namespaces,
    ):
        tai = trustyai_service_with_pvc_storage_multi_ns[ns_index]
        inference_model = gaussian_credit_model_multi_ns[ns_index]
        inference_token = isvc_getter_token_multi_ns[ns_index]

        send_inferences_and_verify_trustyai_service_registered(
            client=admin_client,
            token=current_client_token,
            data_path=f"{DRIFT_BASE_DATA_PATH}/data_batches",
            trustyai_service=tai,
            inference_service=inference_model,
            inference_config=OPENVINO_KSERVE_INFERENCE_CONFIG,
            inference_token=inference_token,
        )

    @pytest.mark.parametrize("ns_index", [0, 1])
    def test_upload_data_to_trustyai_service_multiple_ns(
            self,
            ns_index: int,
            admin_client,
            current_client_token,
            trustyai_service_with_pvc_storage_multi_ns,
            gaussian_credit_model_multi_ns,
            isvc_getter_token_multi_ns,
            model_namespaces,
    ) -> None:
            tai = trustyai_service_with_pvc_storage_multi_ns[ns_index]
            verify_upload_data_to_trustyai_service(
                client=admin_client,
                trustyai_service=tai,
                token=current_client_token,
                data_path=f"{DRIFT_BASE_DATA_PATH}/training_data.json",
            )

    @pytest.mark.parametrize("ns_index", [0, 1])
    def test_drift_metric_schedule_meanshift_multiple_ns(
            self,
            ns_index,
            admin_client,
            current_client_token,
            trustyai_service_with_pvc_storage_multi_ns,
            gaussian_credit_model_multi_ns
    ):
            tai = trustyai_service_with_pvc_storage_multi_ns[ns_index]
            inference_model = gaussian_credit_model_multi_ns[ns_index]

            verify_trustyai_service_metric_scheduling_request(
                client=admin_client,
                trustyai_service=tai,
                token=current_client_token,
                metric_name=TrustyAIServiceMetrics.Drift.MEANSHIFT,
                json_data={
                    "modelId": inference_model.name,
                    "referenceTag": "TRAINING",
                },
            )

    @pytest.mark.parametrize("ns_index", [0, 1])
    def test_drift_metric_delete_multiple_ns(
            self,
            ns_index,
            admin_client,
            current_client_token,
            minio_data_connection_multi_ns,
            trustyai_service_with_pvc_storage_multi_ns
    ):
        tai = trustyai_service_with_pvc_storage_multi_ns[ns_index]
        verify_trustyai_service_metric_delete_request(
                client=admin_client,
                trustyai_service=tai,
                token=current_client_token,
                metric_name=TrustyAIServiceMetrics.Drift.MEANSHIFT,
            )

