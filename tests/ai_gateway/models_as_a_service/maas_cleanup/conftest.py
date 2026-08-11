from collections.abc import Generator

import pytest
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.resource import ResourceEditor

from tests.ai_gateway.models_as_a_service.utils import (
    dsc_uses_aigateway_maas_schema,
    maas_component_patch,
    wait_for_maas_controller_ready,
)
from utilities.constants import DscComponents
from utilities.data_science_cluster_utils import get_dsc_ready_condition, wait_for_dsc_reconciliation


@pytest.fixture(scope="class")
def dsc_with_maas_disabled(
    dsc_resource: DataScienceCluster,
    maas_controller_enabled_latest: DataScienceCluster,
) -> Generator[None]:
    """DSC with MaaS set to Removed using the field supported by the cluster version.

    Restores MaaS to Managed on teardown and waits for the matching readiness condition.
    """
    uses_aigateway_maas_schema = dsc_uses_aigateway_maas_schema(admin_client=dsc_resource.client)
    component_patch = maas_component_patch(
        admin_client=dsc_resource.client,
        models_as_a_service_state=DscComponents.ManagementState.REMOVED,
        aigateway_state=DscComponents.ManagementState.MANAGED,
    )
    baseline_ready_condition = get_dsc_ready_condition(dsc=dsc_resource)
    baseline_time = baseline_ready_condition.get("lastTransitionTime") if baseline_ready_condition else None

    with ResourceEditor(patches={dsc_resource: {"spec": {"components": component_patch}}}):
        wait_for_dsc_reconciliation(dsc=dsc_resource, baseline_time=baseline_time)
        yield

    wait_for_maas_controller_ready(
        dsc_resource=dsc_resource,
        uses_aigateway_maas_schema=uses_aigateway_maas_schema,
    )
