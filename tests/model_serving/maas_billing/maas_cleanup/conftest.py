from collections.abc import Generator

import pytest
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.resource import ResourceEditor

from utilities.constants import DscComponents


@pytest.fixture(scope="class")
def maas_disabled_for_cleanup_test(
    dsc_resource: DataScienceCluster,
    maas_controller_enabled_latest: DataScienceCluster,
) -> Generator[None, None, None]:
    """Patch DSC modelsAsService to Removed, then restore Managed on teardown.

    Depends on maas_controller_enabled_latest so MaaS is guaranteed to be running
    before the disable is triggered, matching the real admin workflow that the bug
    reproduces.
    """
    component_patch = {
        DscComponents.KSERVE: {
            "modelsAsService": {"managementState": DscComponents.ManagementState.REMOVED}
        }
    }
    with ResourceEditor(patches={dsc_resource: {"spec": {"components": component_patch}}}):
        dsc_resource.wait_for_condition(condition="Ready", status="True", timeout=600)
        yield

    dsc_resource.wait_for_condition(condition="ModelsAsServiceReady", status="True", timeout=900)
    dsc_resource.wait_for_condition(condition="Ready", status="True", timeout=600)
