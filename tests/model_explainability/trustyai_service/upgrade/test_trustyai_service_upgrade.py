import pytest

from tests.model_explainability.trustyai_service.service.utils import (
    wait_for_trustyai_db_migration_complete_log,
    patch_trustyai_service_cr,
)
from tests.model_explainability.trustyai_service.constants import DRIFT_BASE_DATA_PATH, TRUSTYAI_DB_MIGRATION_PATCH
from tests.model_explainability.trustyai_service.trustyai_service_utils import (
    send_inferences_and_verify_trustyai_service_registered,
    verify_upload_data_to_trustyai_service,
    verify_trustyai_service_metric_delete_request,
    TrustyAIServiceMetrics,
    verify_trustyai_service_metric_scheduling_request,
)
from utilities.manifests.openvino import OPENVINO_KSERVE_INFERENCE_CONFIG
from timeout_sampler import retry


@pytest.mark.parametrize(
    "model_namespace, trustyai_service",
    [
        pytest.param(
            {"name": "test-trustyaiservice-upgrade"},
            {"storage": "pvc"},
        )
    ],
    indirect=True,
)
class TestPreUpgradeTrustyAIService:
    @pytest.mark.pre_upgrade
    def test_trustyai_service_pre_upgrade_inference(
        self,
        admin_client,
        current_client_token,
        model_namespace,
        isvc_getter_token,
        trustyai_service,
        gaussian_credit_model,
    ) -> None:
        """Set up a TrustyAIService with a model and inference before upgrade."""
        send_inferences_and_verify_trustyai_service_registered(
            client=admin_client,
            token=current_client_token,
            data_path=f"{DRIFT_BASE_DATA_PATH}/data_batches",
            trustyai_service=trustyai_service,
            inference_service=gaussian_credit_model,
            inference_config=OPENVINO_KSERVE_INFERENCE_CONFIG,
            inference_token=isvc_getter_token,
        )

    @pytest.mark.pre_upgrade
    def test_trustyai_service_pre_upgrade_data_upload(
        self,
        admin_client,
        minio_data_connection,
        current_client_token,
        trustyai_service,
    ) -> None:
        """Upload data to TrustyAIService before upgrade."""
        verify_upload_data_to_trustyai_service(
            client=admin_client,
            trustyai_service=trustyai_service,
            token=current_client_token,
            data_path=f"{DRIFT_BASE_DATA_PATH}/training_data.json",
        )

    @pytest.mark.pre_upgrade
    def test_trustyai_service_pre_upgrade_drift_metric_schedule_meanshift(
        self,
        admin_client,
        current_client_token,
        trustyai_service,
        gaussian_credit_model,
    ):
        """Schedule a drift metric before upgrade."""
        verify_trustyai_service_metric_scheduling_request(
            client=admin_client,
            trustyai_service=trustyai_service,
            token=current_client_token,
            metric_name=TrustyAIServiceMetrics.Drift.MEANSHIFT,
            json_data={
                "modelId": "gaussian-credit-model",
                "referenceTag": "TRAINING",
            },
        )


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-trustyaiservice-upgrade"},
        )
    ],
    indirect=True,
)
class TestPostUpgradeTrustyAIService:
    @pytest.mark.post_upgrade
    def test_drift_metric_delete_pre_db_migration(
        self,
        admin_client,
        current_client_token,
        trustyai_service,
    ):
        """Retrieve the metric scheduled before upgrade and delete it."""
        verify_trustyai_service_metric_delete_request(
            client=admin_client,
            trustyai_service=trustyai_service,
            token=current_client_token,
            metric_name=TrustyAIServiceMetrics.Drift.MEANSHIFT,
        )

    @pytest.mark.dependency(name="db_migration")
    @pytest.mark.post_upgrade
    def test_trustyai_service_db_migration(
        self,
        admin_client,
        current_client_token,
        mariadb,
        trustyai_db_ca_secret,
        trustyai_service,
        gaussian_credit_model,
    ) -> None:
        """Verify if TrustyAI DB Migration works as expected.
        This test initializes TrustyAI Service with PVC Storage at first with a database on standby but the service is
         not
        configured to use it.
        Data is uploaded to the PVC, then the TrustyAI CR is patched to trigger a migration from PVC to DB storage.
        config.
        Then waits for the migration success entry in the container logs and patches the service again to remove PVC
        config.
        Finally, a metric is scheduled and checked if the service works as expected post-migration.

        Args:
            admin_client: DynamicClient
            current_client_token: RedactedString
            mariadb: MariaDB
            trustyai_db_ca_secret: None
            trustyai_service: TrustyAIService
            gaussian_credit_model: Generator[InferenceService, Any, Any]

        Returns:
            None
        """
        trustyai_db_migration_patched_service = patch_trustyai_service_cr(
            trustyai_service=trustyai_service, patches=TRUSTYAI_DB_MIGRATION_PATCH
        )

        wait_for_trustyai_db_migration_complete_log(
            client=admin_client,
            trustyai_service=trustyai_db_migration_patched_service,
        )

        @retry(wait_timeout=30, sleep=5)
        def _retry_metric_verify_on_err():
            return bool(
                verify_trustyai_service_metric_scheduling_request(
                    client=admin_client,
                    trustyai_service=trustyai_db_migration_patched_service,
                    token=current_client_token,
                    metric_name=TrustyAIServiceMetrics.Drift.MEANSHIFT,
                    json_data={
                        "modelId": "gaussian-credit-model",
                        "referenceTag": "TRAINING",
                    },
                )
                is None
            )

        _retry_metric_verify_on_err()

    @pytest.mark.dependency(depends=["db_migration"])
    @pytest.mark.post_upgrade
    def test_drift_metric_delete_post_db_migration(
        self,
        admin_client,
        minio_data_connection,
        current_client_token,
        trustyai_db_ca_secret,
        trustyai_service,
    ):
        """Retrieve the metric scheduled post-migration and delete it."""
        verify_trustyai_service_metric_delete_request(
            client=admin_client,
            trustyai_service=trustyai_service,
            token=current_client_token,
            metric_name=TrustyAIServiceMetrics.Drift.MEANSHIFT,
        )
