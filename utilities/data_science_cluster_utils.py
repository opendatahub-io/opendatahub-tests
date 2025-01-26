from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.resource import ResourceEditor
from simple_logger.logger import get_logger

from utilities.constants import DscComponents

LOGGER = get_logger(name=__name__)


@contextmanager
def update_components_in_dsc(
    dsc: DataScienceCluster, components: dict[str, str], wait_for_components_state: bool = True
) -> Generator[DataScienceCluster, Any, Any]:
    """
    Update components in dsc

    Args:
        dsc (DataScienceCluster): DataScienceCluster object
        components (dict[str,dict[str,str]]): Dict of components. key is component name, value: component desired state
        wait_for_components_state (bool): Wait until components state in dsc

    Returns:
        DataScienceCluster

    """
    dsc_dict: dict[str, dict[str, dict[str, dict[str, str]]]] = {}
    dsc_components = dsc.instance.spec.components

    for component_name, desired_state in components.items():
        if dsc_components[component_name].managementState != desired_state:
            dsc_dict.setdefault("spec", {}).setdefault("components", {})[component_name] = {
                "managementState": desired_state
            }
        else:
            LOGGER.warning(f"Component {component_name} was already set to managementState {desired_state}")

    if dsc_dict:
        with ResourceEditor(patches={dsc: dsc_dict}):
            if wait_for_components_state:
                for component in components:
                    dsc.wait_for_condition(condition=DscComponents.COMPONENT_MAPPING[component], status="True")
            yield dsc

        dsc.wait_for_status(status=dsc.Status.READY)

    else:
        yield dsc
