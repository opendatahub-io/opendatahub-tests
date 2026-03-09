from typing import Self

import pytest
from kubernetes.dynamic.client import DynamicClient
from ocp_resources.pod import Pod
from pytest_testconfig import config as py_config
from simple_logger.logger import get_logger

from tests.model_registry.constants import MR_INSTANCE_NAME
from tests.model_registry.utils import wait_for_new_running_mr_pod
from utilities.general import wait_for_container_status

LOGGER = get_logger(name=__name__)


@pytest.mark.usefixtures(
    "updated_dsc_component_state_scope_session", "model_registry_metadata_db_resources", "model_registry_instance"
)
class TestDBMigration:
    @pytest.mark.tier3
    def test_db_migration_negative(
        self: Self,
        admin_client: DynamicClient,
        model_registry_db_instance_pod: Pod,
        set_mr_db_dirty: int,
        model_registry_pod: Pod,
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
        LOGGER.info(f"Model registry pod: {model_registry_pod.name}")
        mr_pod = wait_for_new_running_mr_pod(
            admin_client=admin_client,
            orig_pod_name=model_registry_pod.name,
            namespace=py_config["model_registry_namespace"],
            instance_name=MR_INSTANCE_NAME,
        )
        LOGGER.info(f"Pod that should contains the container in CrashLoopBackOff state: {mr_pod.name}")
        assert wait_for_container_status(mr_pod, "rest-container", Pod.Status.CRASH_LOOPBACK_OFF)

        LOGGER.info("Checking the logs for the expected error")
        log_output = mr_pod.log(container="rest-container")
        expected_error = (
            f"Error: {{{{ALERT}}}} error connecting to datastore: Dirty database version {set_mr_db_dirty}. "
            "Fix and force version."
        )
        assert expected_error in log_output, f"Expected error message not found in logs!\n{log_output}"
