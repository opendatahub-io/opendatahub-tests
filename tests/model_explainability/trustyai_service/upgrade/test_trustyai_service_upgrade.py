import pytest

from tests.model_explainability.trustyai_service.constants import DRIFT_BASE_DATA_PATH
from tests.model_explainability.trustyai_service.trustyai_service_utils import (
    send_inferences_and_verify_trustyai_service_registered,
    verify_upload_data_to_trustyai_service,
    verify_trustyai_service_metric_delete_request,
    TrustyAIServiceMetrics,
    verify_trustyai_service_metric_scheduling_request,
)
from utilities.constants import MinIo
from utilities.manifests.openvino import OPENVINO_KSERVE_INFERENCE_CONFIG


@pytest.mark.parametrize(
    "model_namespace, minio_pod, minio_data_connection",
    [
        pytest.param(
            {"name": "test-trustyaiservice-upgrade"},
            MinIo.PodConfig.MODEL_MESH_MINIO_CONFIG,
            {"bucket": MinIo.Buckets.MODELMESH_EXAMPLE_MODELS},
        )
    ],
    indirect=True,
)
@pytest.mark.usefixtures("minio_pod")
class TestPreUpgradeTrustyAIService:
    @pytest.mark.pre_upgrade
    def test_trustyai_service_pre_upgrade_inference(
        self,
        admin_client,
        current_client_token,
        model_namespace,
        trustyai_service_with_pvc_storage,
        gaussian_credit_model,
    ) -> None:
        """Set up a TrustyAIService with a model and inference before upgrade."""
        send_inferences_and_verify_trustyai_service_registered(
            client=admin_client,
            token=current_client_token,
            data_path=f"{DRIFT_BASE_DATA_PATH}/data_batches",
            trustyai_service=trustyai_service_with_pvc_storage,
            inference_service=gaussian_credit_model,
            inference_config=OPENVINO_KSERVE_INFERENCE_CONFIG,
        )

    @pytest.mark.pre_upgrade
    def test_trustyai_service_pre_upgrade_data_upload(
        self,
        admin_client,
        minio_data_connection,
        current_client_token,
        trustyai_service_with_pvc_storage,
    ) -> None:
        """Upload data to TrustyAIService before upgrade."""
        verify_upload_data_to_trustyai_service(
            client=admin_client,
            trustyai_service=trustyai_service_with_pvc_storage,
            token=current_client_token,
            data_path=f"{DRIFT_BASE_DATA_PATH}/training_data.json",
        )

    @pytest.mark.pre_upgrade
    def test_trustyai_service_pre_upgrade_drift_metric_schedule_meanshift(
        self,
        admin_client,
        current_client_token,
        trustyai_service_with_pvc_storage,
        gaussian_credit_model,
    ):
        """Schedule a drift metric before upgrade."""
        verify_trustyai_service_metric_scheduling_request(
            client=admin_client,
            trustyai_service=trustyai_service_with_pvc_storage,
            token=current_client_token,
            metric_name=TrustyAIServiceMetrics.Drift.MEANSHIFT,
            json_data={
                "modelId": gaussian_credit_model.name,
                "referenceTag": "TRAINING",
            },
        )


@pytest.mark.parametrize(
    "model_namespace, minio_pod, minio_data_connection",
    [
        pytest.param(
            {"name": "test-trustyaiservice-upgrade"},
            MinIo.PodConfig.MODEL_MESH_MINIO_CONFIG,
            {"bucket": MinIo.Buckets.MODELMESH_EXAMPLE_MODELS},
        )
    ],
    indirect=True,
)
@pytest.mark.usefixtures("minio_pod")
class TestPostUpgradeTrustyAIService:
    @pytest.mark.post_upgrade
    def test_drift_metric_delete(
        self,
        admin_client,
        minio_data_connection,
        current_client_token,
        trustyai_service_with_pvc_storage,
    ):
        """Retrieve the metric scheduled before upgrade and delete it."""
        verify_trustyai_service_metric_delete_request(
            client=admin_client,
            trustyai_service=trustyai_service_with_pvc_storage,
            token=current_client_token,
            metric_name=TrustyAIServiceMetrics.Drift.MEANSHIFT,
        )
