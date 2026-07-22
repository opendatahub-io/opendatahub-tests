from collections.abc import Generator

import pytest
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.resource import ResourceEditor

from tests.model_serving.maas_billing.utils import maas_under_aigateway_component_patch
from utilities.constants import DscComponents
from utilities.data_science_cluster_utils import get_dsc_ready_condition, wait_for_dsc_reconciliation


@pytest.fixture(scope="class")
def dsc_with_maas_disabled(
    dsc_resource: DataScienceCluster,
    maas_controller_enabled_latest: DataScienceCluster,
) -> Generator[None]:
    """DSC with aigateway.modelsAsAService set to Removed; aigateway stays Managed.

    Restores nested MaaS to Managed on teardown and waits for AIGatewayReady.
    """
    component_patch = maas_under_aigateway_component_patch(
        models_as_a_service_state=DscComponents.ManagementState.REMOVED,
        aigateway_state=DscComponents.ManagementState.MANAGED,
    )
    baseline_ready_condition = get_dsc_ready_condition(dsc=dsc_resource)
    baseline_time = baseline_ready_condition.get("lastTransitionTime") if baseline_ready_condition else None

    with ResourceEditor(patches={dsc_resource: {"spec": {"components": component_patch}}}):
        wait_for_dsc_reconciliation(dsc=dsc_resource, baseline_time=baseline_time)
        yield

    dsc_resource.wait_for_condition(
        condition=DscComponents.ConditionType.AIGATEWAY_READY,
        status="True",
        timeout=900,
    )
    dsc_resource.wait_for_condition(condition="Ready", status="True", timeout=600)
