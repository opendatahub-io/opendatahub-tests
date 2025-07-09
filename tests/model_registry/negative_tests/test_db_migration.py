import pytest
from typing import Self
from simple_logger.logger import get_logger
from ocp_resources.model_registry_modelregistry_opendatahub_io import ModelRegistry
from pytest_testconfig import config as py_config
from ocp_resources.pod import Pod
from utilities.constants import DscComponents
from tests.model_registry.constants import MR_INSTANCE_NAME
from kubernetes.dynamic.client import DynamicClient
from utilities.general import wait_for_pods_by_labels


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
@pytest.mark.usefixtures("updated_dsc_component_state_scope_class")
class TestDBMigration:
    def test_db_migration_negative(
        self: Self,
        admin_client: DynamicClient,
        model_registry_instance: ModelRegistry,
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
        mr_pods = wait_for_pods_by_labels(
            admin_client=admin_client,
            namespace=py_config["model_registry_namespace"],
            label_selector=f"app={MR_INSTANCE_NAME}",
            expected_num_pods=1,
        )
        mr_pod = mr_pods[0]
        LOGGER.info("Checking the logs for the expected error")

        log_output = mr_pod.log(container="rest-container")
        expected_error = (
            f"Error: {{{{ALERT}}}} error connecting to datastore: Dirty database version {set_mr_db_dirty}. "
            "Fix and force version."
        )
        assert expected_error in log_output, "Expected error message not found in logs!"
