import pytest
from typing import Self
from simple_logger.logger import get_logger
from pytest_testconfig import config as py_config

from ocp_resources.pod import Pod
from utilities.constants import DscComponents
from tests.model_registry.constants import MR_INSTANCE_NAME, MODEL_REGISTRY_DB_SECRET_STR_DATA
from kubernetes.dynamic.client import DynamicClient
from utilities.general import wait_for_pods_by_labels, wait_for_container_status


LOGGER = get_logger(name=__name__)


@pytest.mark.parametrize(
    "updated_dsc_component_state_scope_class",
    [
        pytest.param({
            "component_patch": {
                DscComponents.MODELREGISTRY: {
                    "managementState": DscComponents.ManagementState.MANAGED,
                    "registriesNamespace": py_config["model_registry_namespace"],
                },
            }
        }),
    ],
    indirect=True,
)
@pytest.mark.usefixtures(
    "updated_dsc_component_state_scope_class", "model_registry_mysql_metadata_db", "model_registry_instance_mysql"
)
class TestDBMigration:
    @pytest.mark.smoke
    def test_db_migration_negative(
        self: Self,
        admin_client: DynamicClient,
        model_registry_db_instance_pod: Pod,
        set_mr_db_dirty: int,
        delete_mr_deployment: None,
    ):
        """
        RHOAIENG-27505: This test is to check the migration error when the database is dirty.
        The test will:
        1. Set the dirty flag to 1 for the latest migration version
        2. Delete the model registry deployment
        3. Check the logs for the expected error
        """

        # First, let's see what migration versions exist in the database
        LOGGER.info("Checking all migration versions in the database")
        all_migrations_output = model_registry_db_instance_pod.execute(
            command=[
                "mysql",
                "-u",
                MODEL_REGISTRY_DB_SECRET_STR_DATA["database-user"],
                f"-p{MODEL_REGISTRY_DB_SECRET_STR_DATA['database-password']}",
                "-e",
                "SELECT version, dirty FROM schema_migrations ORDER BY version DESC;",
                "model_registry",
            ]
        )
        LOGGER.info(f"All migrations in database:\n{all_migrations_output}")

        # Verify that the dirty flag was set correctly
        LOGGER.info(f"Verifying dirty flag was set for migration version {set_mr_db_dirty}")
        verification_output = model_registry_db_instance_pod.execute(
            command=[
                "mysql",
                "-u",
                MODEL_REGISTRY_DB_SECRET_STR_DATA["database-user"],
                f"-p{MODEL_REGISTRY_DB_SECRET_STR_DATA['database-password']}",
                "-e",
                f"SELECT version, dirty FROM schema_migrations WHERE version = {set_mr_db_dirty};",
                "model_registry",
            ]
        )
        LOGGER.info(f"Database verification output: {verification_output}")

        # Parse the output to check if dirty flag is set
        lines = verification_output.strip().split("\n")
        if len(lines) >= 2:
            # Skip header line, check data line
            data_line = lines[1].strip()
            if data_line:
                parts = data_line.split("\t")
                if len(parts) >= 2:
                    version = parts[0]
                    dirty_flag = parts[1]
                    LOGGER.info(f"Migration version {version} has dirty flag set to: {dirty_flag}")
                    if dirty_flag != "1":
                        LOGGER.error(f"ERROR: Dirty flag is not set to 1! Found: {dirty_flag}")
                        assert False, f"Dirty flag should be 1, but found {dirty_flag}"
                    else:
                        LOGGER.info("✓ Dirty flag is correctly set to 1")
                else:
                    LOGGER.error(f"Unexpected data format: {data_line}")
                    assert False, f"Unexpected data format: {data_line}"
            else:
                LOGGER.error("No data found for the specified migration version")
                assert False, "No data found for the specified migration version"
        else:
            LOGGER.error(f"Unexpected output format: {verification_output}")
            assert False, f"Unexpected output format: {verification_output}"
        # Wait for the new pod to be created after deployment deletion
        LOGGER.info("Waiting for new model registry pod to be created after deployment deletion")

        # Verify that the dirty flag persists after deployment deletion
        LOGGER.info("Verifying dirty flag persists after deployment deletion")
        persistence_verification = model_registry_db_instance_pod.execute(
            command=[
                "mysql",
                "-u",
                MODEL_REGISTRY_DB_SECRET_STR_DATA["database-user"],
                f"-p{MODEL_REGISTRY_DB_SECRET_STR_DATA['database-password']}",
                "-e",
                f"SELECT version, dirty FROM schema_migrations WHERE version = {set_mr_db_dirty};",
                "model_registry",
            ]
        )
        LOGGER.info(f"Persistence verification output: {persistence_verification}")

        # Check if dirty flag still persists
        lines = persistence_verification.strip().split("\n")
        if len(lines) >= 2:
            data_line = lines[1].strip()
            if data_line:
                parts = data_line.split("\t")
                if len(parts) >= 2:
                    dirty_flag = parts[1]
                    if dirty_flag == "1":
                        LOGGER.info("✓ Dirty flag persists correctly after deployment deletion")
                    else:
                        LOGGER.error(f"ERROR: Dirty flag no longer persists! Found: {dirty_flag}")
                        assert False, f"Dirty flag should persist as 1, but found {dirty_flag}"
                else:
                    LOGGER.error(f"Unexpected persistence data format: {data_line}")
                    assert False, f"Unexpected persistence data format: {data_line}"
            else:
                LOGGER.error("No data found during persistence check")
                assert False, "No data found during persistence check"
        else:
            LOGGER.error(f"Unexpected persistence output format: {persistence_verification}")
            assert False, f"Unexpected persistence output format: {persistence_verification}"
        # Get all pods and log them for debugging
        all_pods = list(
            Pod.get(
                dyn_client=admin_client,
                namespace=py_config["model_registry_namespace"],
                label_selector=f"app={MR_INSTANCE_NAME}",
            )
        )
        LOGGER.info(f"All MR pods found: {[pod.name for pod in all_pods]}")

        mr_pods = wait_for_pods_by_labels(
            admin_client=admin_client,
            namespace=py_config["model_registry_namespace"],
            label_selector=f"app={MR_INSTANCE_NAME}",
            expected_num_pods=1,
        )
        mr_pod = mr_pods[0]
        LOGGER.info(f"Selected pod: {mr_pod.name}")

        # Verify the pod still exists before checking status
        try:
            mr_pod.instance
            LOGGER.info(f"Pod {mr_pod.name} exists and is accessible")
        except Exception as e:
            LOGGER.error(f"Pod {mr_pod.name} is not accessible: {e}")
            raise

        # Final verification: Check if dirty flag is still present after pod creation
        LOGGER.info("Final verification: Checking if dirty flag is still present after pod creation")
        final_verification = model_registry_db_instance_pod.execute(
            command=[
                "mysql",
                "-u",
                MODEL_REGISTRY_DB_SECRET_STR_DATA["database-user"],
                f"-p{MODEL_REGISTRY_DB_SECRET_STR_DATA['database-password']}",
                "-e",
                f"SELECT version, dirty FROM schema_migrations WHERE version = {set_mr_db_dirty};",
                "model_registry",
            ]
        )
        LOGGER.info(f"Final verification output: {final_verification}")

        # Check the rest-container logs to see what's happening with the Model Registry
        LOGGER.info("Checking rest-container logs to understand Model Registry state")
        try:
            log_output = mr_pod.log(container="rest-container")
            LOGGER.info(f"Model Registry logs:\n{log_output}")

            # Check for the expected error message in the logs
            expected_error = (
                f"Error: {{{{ALERT}}}} error connecting to datastore: Dirty database version {set_mr_db_dirty}. "
                "Fix and force version."
            )
            if expected_error in log_output:
                LOGGER.info("✓ Found expected error message in logs")
            else:
                LOGGER.warning("Expected error message not found in logs")
                if "dirty" in log_output.lower() or "migration" in log_output.lower():
                    LOGGER.info("Found migration-related messages in logs, but not the exact expected error")
                else:
                    LOGGER.warning("No migration-related messages found in logs")

        except Exception as e:
            LOGGER.warning(f"Could not get rest-container logs: {e}")

        # Now check if the rest-container is in CrashLoopBackOff state
        LOGGER.info("Checking if rest-container is in CrashLoopBackOff state")
        try:
            # Wait for the pod to be in CrashLoopBackOff state
            assert wait_for_container_status(mr_pod, "rest-container", Pod.Status.CRASH_LOOPBACK_OFF)
            LOGGER.info("✓ rest-container is in CrashLoopBackOff state as expected")
        except Exception as e:
            LOGGER.error(f"rest-container is not in CrashLoopBackOff state: {e}")
            LOGGER.error("This suggests the dirty database flag is not causing the Model Registry to crash")
            assert False, "rest-container should be in CrashLoopBackOff state due to dirty database"
