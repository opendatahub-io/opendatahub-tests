from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.resource import ResourceEditor
from simple_logger.logger import get_logger
from timeout_sampler import retry

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
    component_to_reconcile = {}

    for component_name, desired_state in components.items():
        if (orig_state := dsc_components[component_name].managementState) != desired_state:
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
                    dsc.wait_for_condition(condition=DscComponents.COMPONENT_MAPPING[component], status="True")
            yield dsc

        for component, state in component_to_reconcile.items():
            if state == DscComponents.ManagementState.MANAGED:
                dsc.wait_for_condition(condition=DscComponents.COMPONENT_MAPPING[component], status="True")

    else:
        yield dsc


def get_dsc_ready_condition(dsc: DataScienceCluster) -> dict[str, Any] | None:
    """Get DSC Ready condition.

    Args:
        dsc: DataScienceCluster resource

    Returns:
        The Ready condition dict (with 'status', 'lastTransitionTime', etc.), or None if not found
    """
    return next(
        (
            condition
            for condition in dsc.instance.status.conditions or []
            if condition.type == DataScienceCluster.Condition.READY
        ),
        None,
    )


@retry(wait_timeout=300, sleep=5)
def wait_for_dsc_reconciliation(dsc: DataScienceCluster, baseline_time: str | None) -> bool:
    """Wait for DSC to reconcile after a ResourceEditor patch.

    This function prevents false positives where DSC reports Ready=True immediately
    after a patch, before actual reconciliation begins. It waits for:
    1. lastTransitionTime to change (reconciliation started)
    2. Ready=True condition (reconciliation completed)

    Args:
        dsc: DataScienceCluster resource
        baseline_time: The Ready condition lastTransitionTime before the patch, or None if not found

    Returns:
        True when DSC has reconciled and is Ready
    """
    ready_condition = get_dsc_ready_condition(dsc=dsc)
    current_time = ready_condition.get("lastTransitionTime") if ready_condition else None
    dsc_reconciling = current_time != baseline_time
    dsc_ready = ready_condition and ready_condition.get("status") == DataScienceCluster.Condition.Status.TRUE

    # Still waiting for reconciliation to start (timestamp unchanged)
    if not dsc_reconciling:
        LOGGER.info(f"Waiting for DSC reconciliation to start (baseline: {baseline_time or 'None'})...")
        return False

    # Timestamp changed but DSC is not Ready yet
    if not dsc_ready:
        LOGGER.info(f"DSC reconciliation in progress (timestamp: {current_time or 'None'}), waiting for Ready=True...")
        return False

    # DSC Reconciled: timestamp changed AND Ready=True
    LOGGER.info(
        f"DSC reconciliation complete: timestamp changed from {baseline_time or 'None'} "
        f"to {current_time or 'None'} and Ready=True"
    )
    return True
