from typing import Any, Generator

import pytest
from ocp_resources.data_science_cluster import DataScienceCluster

from utilities.data_science_cluster_utils import update_components_in_dsc
from utilities.constants import DscComponents


@pytest.fixture(scope="class")
def managed_modelmesh_kserve_in_dsc(
    dsc_resource: DataScienceCluster,
) -> Generator[DataScienceCluster, Any, Any]:
    with update_components_in_dsc(
        dsc=dsc_resource,
        components={
            DscComponents.MODELMESHSERVING: DscComponents.ManagementState.MANAGED,
            DscComponents.KSERVE: DscComponents.ManagementState.MANAGED,
        },
        wait_for_components_state=False,
    ) as dsc:
        yield dsc
