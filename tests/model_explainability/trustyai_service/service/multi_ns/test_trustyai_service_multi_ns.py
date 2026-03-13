import pytest

from tests.model_explainability.trustyai_service.constants import DRIFT_BASE_DATA_PATH
from tests.model_explainability.trustyai_service.trustyai_service_utils import (
    TrustyAIServiceMetrics,
    send_inferences_and_verify_trustyai_service_registered,
    verify_trustyai_service_metric_delete_request,
    verify_trustyai_service_metric_scheduling_request,
    verify_upload_data_to_trustyai_service,
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
            ],
        ),
    ],
    indirect=True,
)
@pytest.mark.rawdeployment
class TestTrustyAIServiceMultipleNS:
    """Verifies TrustyAIService operations across multiple namespaces,
    i.e. registering inference requests, uploading data, scheduling and deleting metrics,
    that can be performed with a TrustyAIService metric(drift, in this case)"""

    def test_drift_send_inference_and_verify_trustyai_service_multiple_ns(
        self,
        admin_client,
        current_client_token,
        trustyai_service_with_pvc_storage_multi_ns,
        gaussian_credit_model_multi_ns,
        isvc_getter_token_multi_ns,
    ):
        for tai, model, token in zip(
            trustyai_service_with_pvc_storage_multi_ns, gaussian_credit_model_multi_ns, isvc_getter_token_multi_ns
        ):
            send_inferences_and_verify_trustyai_service_registered(
                client=admin_client,
                token=current_client_token,
                data_path=f"{DRIFT_BASE_DATA_PATH}/data_batches",
                trustyai_service=tai,
                inference_service=model,
                inference_config=OPENVINO_KSERVE_INFERENCE_CONFIG,
                inference_token=token,
            )

    def test_upload_data_to_trustyai_service_multiple_ns(
        self,
        admin_client,
        current_client_token,
        trustyai_service_with_pvc_storage_multi_ns,
        gaussian_credit_model_multi_ns,
        isvc_getter_token_multi_ns,
        model_namespaces,
    ) -> None:
        for tai in trustyai_service_with_pvc_storage_multi_ns:
            verify_upload_data_to_trustyai_service(
                client=admin_client,
                trustyai_service=tai,
                token=current_client_token,
                data_path=f"{DRIFT_BASE_DATA_PATH}/training_data.json",
            )

    def test_drift_metric_schedule_meanshift_multiple_ns(
        self,
        admin_client,
        current_client_token,
        trustyai_service_with_pvc_storage_multi_ns,
        gaussian_credit_model_multi_ns,
    ):
        for tai, inference_model in zip(trustyai_service_with_pvc_storage_multi_ns, gaussian_credit_model_multi_ns):
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

    def test_drift_metric_delete_multiple_ns(
        self,
        admin_client,
        current_client_token,
        minio_data_connection_multi_ns,
        trustyai_service_with_pvc_storage_multi_ns,
    ):
        for tai in trustyai_service_with_pvc_storage_multi_ns:
            verify_trustyai_service_metric_delete_request(
                client=admin_client,
                trustyai_service=tai,
                token=current_client_token,
                metric_name=TrustyAIServiceMetrics.Drift.MEANSHIFT,
            )


@pytest.mark.parametrize(
    "model_namespaces, minio_pod, minio_data_connection_multi_ns",
    [
        pytest.param(
            [
                {"name": "test-trustyaiservice-multins-1"},
                {"name": "test-trustyaiservice-multins-2"},
            ],
            MinIo.PodConfig.MODEL_MESH_MINIO_CONFIG,
            [{"bucket": MinIo.Buckets.MODELMESH_EXAMPLE_MODELS}] * 2,
        ),
    ],
    indirect=True,
)
@pytest.mark.usefixtures("minio_pod")
@pytest.mark.rawdeployment
class TestDriftMetricsWithDBStorageMultiNs:
    """
    Verifies drift metric functionality in TrustyAI using MariaDB storage
    for multiple namespaces with OpenVINO KServe models.
    """

    def test_drift_send_inference_and_verify_trustyai_service_with_db_storage(
        self,
        admin_client,
        current_client_token,
        model_namespaces,
        trustyai_service_with_db_storage_multi_ns,
        gaussian_credit_model_multi_ns,
        isvc_getter_token_multi_ns,
    ):
        for tai, inference_model, inference_token in zip(
            trustyai_service_with_db_storage_multi_ns, gaussian_credit_model_multi_ns, isvc_getter_token_multi_ns
        ):
            send_inferences_and_verify_trustyai_service_registered(
                client=admin_client,
                token=current_client_token,
                data_path=f"{DRIFT_BASE_DATA_PATH}/data_batches",
                trustyai_service=tai,
                inference_service=inference_model,
                inference_config=OPENVINO_KSERVE_INFERENCE_CONFIG,
                inference_token=inference_token,
            )

    def test_upload_data_to_trustyai_service_with_db_storage(
        self,
        admin_client,
        current_client_token,
        trustyai_service_with_db_storage_multi_ns,
        minio_data_connection_multi_ns,
    ):
        for tai in trustyai_service_with_db_storage_multi_ns:
            verify_upload_data_to_trustyai_service(
                client=admin_client,
                trustyai_service=tai,
                token=current_client_token,
                data_path=f"{DRIFT_BASE_DATA_PATH}/training_data.json",
            )

    def test_drift_metric_schedule_with_db_storage(
        self,
        admin_client,
        current_client_token,
        trustyai_service_with_db_storage_multi_ns,
        gaussian_credit_model_multi_ns,
    ):
        for tai, model in zip(trustyai_service_with_db_storage_multi_ns, gaussian_credit_model_multi_ns):
            verify_trustyai_service_metric_scheduling_request(
                client=admin_client,
                trustyai_service=tai,
                token=current_client_token,
                metric_name=TrustyAIServiceMetrics.Drift.MEANSHIFT,
                json_data={
                    "modelId": model.name,
                    "referenceTag": "TRAINING",
                },
            )

    def test_drift_metric_delete_with_db_storage(
        self,
        admin_client,
        minio_data_connection_multi_ns,
        current_client_token,
        trustyai_service_with_db_storage_multi_ns,
    ):
        for tai in trustyai_service_with_db_storage_multi_ns:
            verify_trustyai_service_metric_delete_request(
                client=admin_client,
                trustyai_service=tai,
                token=current_client_token,
                metric_name=TrustyAIServiceMetrics.Drift.MEANSHIFT,
            )
