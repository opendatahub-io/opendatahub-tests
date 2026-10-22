import pytest
from ocp_resources.deployment import Deployment
from timeout_sampler import retry

from tests.ai_safety.trustyai_service.constants import DRIFT_BASE_DATA_PATH, TRUSTYAI_DB_MIGRATION_PATCH
from tests.ai_safety.trustyai_service.service.utils import (
    patch_trustyai_service_cr,
    prepare_trustyai_db_migration,
    stage_trustyai_legacy_migration_file,
    wait_for_trustyai_db_migration_rollout,
)
from tests.ai_safety.trustyai_service.trustyai_service_utils import (
    TrustyAIServiceMetrics,
    get_num_observations_from_trustyai_service,
    send_inferences_and_verify_trustyai_service_registered,
    verify_trustyai_service_metric_delete_request,
    verify_trustyai_service_metric_registered_post_upgrade,
    verify_trustyai_service_metric_scheduling_request,
    verify_upload_data_to_trustyai_service,
)
from tests.ai_safety.trustyai_service.utils import validate_db_credentials_secret
from utilities.manifests.openvino import OPENVINO_KSERVE_INFERENCE_CONFIG


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
    """Post-upgrade tests for TrustyAI Service verifying PVC-to-DB storage migration."""

    @pytest.mark.dependency(name="pvc_metric_re_registered")
    @pytest.mark.post_upgrade
    def test_trustyai_service_post_upgrade_preexisting_metric_re_registered(
        self,
        admin_client,
        current_client_token,
        trustyai_service,
    ) -> None:
        """Verify the drift metric scheduled pre-upgrade was re-registered by TrustyAI after upgrade."""
        verify_trustyai_service_metric_registered_post_upgrade(
            client=admin_client,
            trustyai_service=trustyai_service,
            token=current_client_token,
            metric_name=TrustyAIServiceMetrics.Drift.MEANSHIFT,
        )

    @pytest.mark.dependency(depends=["pvc_metric_re_registered"])
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
        """Given pre-upgrade PVC observations, when storage changes to DATABASE, then data and metrics survive."""
        expected_observations = get_num_observations_from_trustyai_service(
            client=admin_client,
            token=current_client_token,
            trustyai_service=trustyai_service,
            model_name="gaussian-credit-model",
        )
        assert expected_observations > 0, "Training observations were not present before database migration"
        stage_trustyai_legacy_migration_file(
            client=admin_client,
            namespace=trustyai_service.namespace,
            service_name=trustyai_service.name,
            model_name="gaussian-credit-model",
        )
        trustyai_deployment = Deployment(
            client=admin_client,
            name=trustyai_service.name,
            namespace=trustyai_service.namespace,
            ensure_exists=True,
        )
        original_deployment_spec = trustyai_deployment.instance.spec.template.spec.to_dict()
        trustyai_db_migration_patched_service = patch_trustyai_service_cr(
            trustyai_service=trustyai_service, patches=TRUSTYAI_DB_MIGRATION_PATCH
        )

        prepare_trustyai_db_migration(
            client=admin_client,
            namespace=trustyai_service.namespace,
            service_name=trustyai_service.name,
            original_deployment_spec=original_deployment_spec,
        )
        wait_for_trustyai_db_migration_rollout(
            client=admin_client,
            namespace=trustyai_service.namespace,
            service_name=trustyai_service.name,
        )

        migrated_observations = get_num_observations_from_trustyai_service(
            client=admin_client,
            token=current_client_token,
            trustyai_service=trustyai_db_migration_patched_service,
            model_name="gaussian-credit-model",
        )
        assert migrated_observations == expected_observations, (
            "Database migration changed the gaussian-credit-model observation count: "
            f"expected {expected_observations}, got {migrated_observations}"
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


@pytest.mark.parametrize(
    "model_namespace, trustyai_service",
    [
        pytest.param(
            {"name": "test-trustyaiservice-db-upgrade"},
            {"storage": "db"},
        )
    ],
    indirect=True,
)
@pytest.mark.rawdeployment
class TestPreUpgradeTrustyAIServiceDB:
    """Pre-upgrade tests for TrustyAI Service with DB storage from the start."""

    @pytest.mark.dependency(name="db_pre_upgrade_inference")
    @pytest.mark.pre_upgrade
    def test_trustyai_service_db_pre_upgrade_inference(
        self,
        admin_client,
        current_client_token,
        model_namespace,
        isvc_getter_token,
        trustyai_service,
        gaussian_credit_model,
    ) -> None:
        """Send inference data and verify observations are registered by TrustyAI with DB storage."""
        send_inferences_and_verify_trustyai_service_registered(
            client=admin_client,
            token=current_client_token,
            data_path=f"{DRIFT_BASE_DATA_PATH}/data_batches",
            trustyai_service=trustyai_service,
            inference_service=gaussian_credit_model,
            inference_config=OPENVINO_KSERVE_INFERENCE_CONFIG,
            inference_token=isvc_getter_token,
        )

    @pytest.mark.dependency(name="db_pre_upgrade_data_upload", depends=["db_pre_upgrade_inference"])
    @pytest.mark.pre_upgrade
    def test_trustyai_service_db_pre_upgrade_data_upload(
        self,
        admin_client,
        current_client_token,
        trustyai_service,
    ) -> None:
        """Upload reference/training data to TrustyAI service configured with DB storage and verify it exists."""
        verify_upload_data_to_trustyai_service(
            client=admin_client,
            trustyai_service=trustyai_service,
            token=current_client_token,
            data_path=f"{DRIFT_BASE_DATA_PATH}/training_data.json",
        )

    @pytest.mark.dependency(depends=["db_pre_upgrade_data_upload"])
    @pytest.mark.pre_upgrade
    def test_trustyai_service_db_pre_upgrade_drift_metric_schedule_meanshift(
        self,
        admin_client,
        current_client_token,
        trustyai_service,
        gaussian_credit_model,
    ) -> None:
        """Schedule a mean shift drift metric to be calculated before upgrade so it can be verified post-upgrade."""
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
    "model_namespace, trustyai_service",
    [
        pytest.param(
            {"name": "test-trustyaiservice-db-upgrade"},
            {"storage": "db"},
        )
    ],
    indirect=True,
)
@pytest.mark.rawdeployment
class TestPostUpgradeTrustyAIServiceDB:
    """Post-upgrade tests for TrustyAI Service with DB storage persistence."""

    @pytest.mark.dependency(name="db_credentials_check")
    @pytest.mark.post_upgrade
    def test_trustyai_service_db_credentials_survived_upgrade(
        self,
        admin_client,
        model_namespace,
        trustyai_service,
        db_credentials_secret,
        mariadb,
    ) -> None:
        """Verify db-credentials secret survived the upgrade process."""
        # Verify the secret still exists after upgrade (using fixture)
        assert db_credentials_secret.exists, (
            f"db-credentials secret does not exist in namespace {model_namespace.name} after upgrade. "
            f"This indicates the secret was deleted during the upgrade process."
        )

        # Verify the secret still has all required keys
        validate_db_credentials_secret(secret=db_credentials_secret, namespace_name=model_namespace.name)

        # Verify MariaDB still exists (using fixture)
        assert mariadb.exists, (
            f"MariaDB instance does not exist in namespace {model_namespace.name} after upgrade. "
            f"Database was likely deleted during upgrade."
        )

    @pytest.mark.dependency(name="db_metric_re_registered", depends=["db_credentials_check"])
    @pytest.mark.post_upgrade
    def test_trustyai_service_db_post_upgrade_preexisting_metric_re_registered(
        self,
        admin_client,
        current_client_token,
        trustyai_db_ca_secret,
        trustyai_service,
    ) -> None:
        """Verify the drift metric scheduled pre-upgrade was re-registered by TrustyAI after upgrade."""
        verify_trustyai_service_metric_registered_post_upgrade(
            client=admin_client,
            trustyai_service=trustyai_service,
            token=current_client_token,
            metric_name=TrustyAIServiceMetrics.Drift.MEANSHIFT,
        )

    @pytest.mark.dependency(name="db_verify_persistence", depends=["db_metric_re_registered"])
    @pytest.mark.post_upgrade
    def test_trustyai_service_db_post_upgrade_preexisting_metric_can_be_deleted(
        self,
        admin_client,
        current_client_token,
        trustyai_db_ca_secret,
        trustyai_service,
    ) -> None:
        """Verify drift metric scheduled pre-upgrade still exists after upgrade by deleting it."""
        verify_trustyai_service_metric_delete_request(
            client=admin_client,
            trustyai_service=trustyai_service,
            token=current_client_token,
            metric_name=TrustyAIServiceMetrics.Drift.MEANSHIFT,
        )

    @pytest.mark.dependency(depends=["db_verify_persistence"])
    @pytest.mark.post_upgrade
    def test_trustyai_service_db_post_upgrade_new_metric_schedule_and_cleanup(
        self,
        admin_client,
        current_client_token,
        trustyai_db_ca_secret,
        trustyai_service,
        gaussian_credit_model,
    ) -> None:
        """Schedule a new metric after upgrade and clean it up to verify DB storage works post-upgrade."""
        verify_trustyai_service_metric_scheduling_request(
            client=admin_client,
            trustyai_service=trustyai_service,
            token=current_client_token,
            metric_name=TrustyAIServiceMetrics.Drift.KSTEST,
            json_data={
                "modelId": "gaussian-credit-model",
                "referenceTag": "TRAINING",
            },
        )

        verify_trustyai_service_metric_delete_request(
            client=admin_client,
            trustyai_service=trustyai_service,
            token=current_client_token,
            metric_name=TrustyAIServiceMetrics.Drift.KSTEST,
        )
