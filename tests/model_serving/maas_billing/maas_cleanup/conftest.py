from collections.abc import Generator

import pytest
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.resource import ResourceEditor

from utilities.constants import DscComponents
from utilities.data_science_cluster_utils import get_dsc_ready_condition, wait_for_dsc_reconciliation


@pytest.fixture(scope="class")
def maas_disabled_for_cleanup_test(
    dsc_resource: DataScienceCluster,
    maas_controller_enabled_latest: DataScienceCluster,
) -> Generator[None]:
    """Patch DSC modelsAsService to Removed, then restore Managed on teardown."""
    component_patch = {
        DscComponents.KSERVE: {"modelsAsService": {"managementState": DscComponents.ManagementState.REMOVED}}
    }
    baseline_ready_condition = get_dsc_ready_condition(dsc=dsc_resource)
    baseline_time = baseline_ready_condition.get("lastTransitionTime") if baseline_ready_condition else None

    with ResourceEditor(patches={dsc_resource: {"spec": {"components": component_patch}}}):
        wait_for_dsc_reconciliation(dsc=dsc_resource, baseline_time=baseline_time)
        yield

    dsc_resource.wait_for_condition(condition="ModelsAsServiceReady", status="True", timeout=900)
    dsc_resource.wait_for_condition(condition="Ready", status="True", timeout=600)
