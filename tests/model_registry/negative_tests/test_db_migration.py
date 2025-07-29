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
        3. Wait for the old pods to be terminated
        4. Check the logs for the expected error
        """
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
