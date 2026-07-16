from contextlib import contextmanager
from typing import Any, Generator

from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.resource import ResourceEditor
from simple_logger.logger import get_logger
from timeout_sampler import retry

from utilities.constants import DscComponents, Timeout

LOGGER = get_logger(name=__name__)


def _get_component_spec_management_state(dsc: DataScienceCluster, component_name: str) -> str | None:
    """Read a component managementState from DSC spec only."""
    spec_component = dsc.instance.spec.components.get(component_name)
    if spec_component:
        return getattr(spec_component, "managementState", None)
    return None


@contextmanager
def update_components_in_dsc(
    dsc: DataScienceCluster,
    components: dict[str, str],
    wait_for_components_state: bool = True,
    condition_wait_timeout: int = Timeout.TIMEOUT_5MIN,
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
    component_to_reconcile = {}

    for component_name, desired_state in components.items():
        orig_state = _get_component_spec_management_state(dsc=dsc, component_name=component_name)
        if orig_state != desired_state:
            dsc_dict.setdefault("spec", {}).setdefault("components", {})[component_name] = {
                "managementState": desired_state
            }
            component_to_reconcile[component_name] = orig_state
        else:
            LOGGER.warning(f"Component {component_name} was already set to managementState {desired_state}")

    if dsc_dict:
        with ResourceEditor(patches={dsc: dsc_dict}):
            if wait_for_components_state:
                for component in components:
                    dsc.wait_for_condition(
                        condition=DscComponents.COMPONENT_MAPPING[component],
                        status="True",
                        timeout=condition_wait_timeout,
                    )
            yield dsc

        for component, state in component_to_reconcile.items():
            if state == DscComponents.ManagementState.MANAGED:
                dsc.wait_for_condition(
                    condition=DscComponents.COMPONENT_MAPPING[component],
                    status="True",
                    timeout=condition_wait_timeout,
                )

    else:
        yield dsc