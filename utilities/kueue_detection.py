"""
Utility functions to detect Kueue installation scenarios and determine appropriate test setup.
"""

from enum import Enum
from typing import Optional

from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.cluster_service_version import ClusterServiceVersion
from simple_logger.logger import get_logger

from utilities.constants import DscComponents
from utilities.infra import get_data_science_cluster

LOGGER = get_logger(name=__name__)


class KueueInstallationScenario(Enum):
    """
    Enum for different Kueue installation scenarios
    """

    # RHOAI Kueue: No Red Hat build operator, DSC Kueue component = Managed
    RHOAI_MANAGED = "rhoai_managed"

    # Red Hat Kueue: Red Hat build of Kueue operator present, DSC Kueue component = Unmanaged
    RHOAI_UNMANAGED = "rhoai_unmanaged"

    # Unknown/unsupported scenario
    UNKNOWN = "unknown"


def is_redhat_kueue_operator_installed(client: DynamicClient) -> bool:
    """
    Check if Red Hat build of Kueue operator is installed on the cluster.

    Args:
        client: DynamicClient to interact with the cluster

    Returns:
        bool: True if Red Hat build of Kueue operator is installed, False otherwise
    """
    try:
        # Look for Red Hat build of Kueue operator CSV in openshift-operators namespace
        csvs = ClusterServiceVersion.get(dyn_client=client, namespace="openshift-operators")
        for csv in csvs:
            # Check for Red Hat build of Kueue operator CSV with name pattern kueue-operator.*
            if csv.name.startswith("kueue-operator"):
                LOGGER.info(f"Found Red Hat build of Kueue operator CSV: {csv.name}")
                return True

        LOGGER.info("No Red Hat build of Kueue operator CSV found (no CSV with name starting with 'kueue-operator')")
        return False

    except ResourceNotFoundError:
        LOGGER.info("No ClusterServiceVersions found in openshift-operators namespace")
        return False
    except Exception as e:
        LOGGER.warning(f"Error checking for Red Hat build of Kueue operator: {e}")
        return False


def get_dsc_kueue_management_state(client: DynamicClient) -> Optional[str]:
    """
    Get the Kueue component management state from DataScienceCluster.

    Args:
        client: DynamicClient to interact with the cluster

    Returns:
        Optional[str]: Management state of Kueue component or None if not found
    """
    try:
        dsc = get_data_science_cluster(client=client)
        dsc_components = dsc.instance.spec.components

        # Check if kueue component exists in DSC
        if hasattr(dsc_components, DscComponents.KUEUE):
            kueue_state = dsc_components.kueue.managementState
            LOGGER.info(f"Found Kueue component in DSC with managementState: {kueue_state}")
            return kueue_state
        else:
            LOGGER.info("No Kueue component found in DataScienceCluster")
            return None

    except Exception as e:
        LOGGER.warning(f"Error getting DSC Kueue management state: {e}")
        return None


def detect_kueue_installation_scenario(client: DynamicClient) -> KueueInstallationScenario:
    """
    Detect the current Kueue installation scenario on the cluster.

    Args:
        client: DynamicClient to interact with the cluster

    Returns:
        KueueInstallationScenario: The detected installation scenario
    """
    rh_kueue_operator_installed = is_redhat_kueue_operator_installed(client=client)
    dsc_kueue_state = get_dsc_kueue_management_state(client=client)

    LOGGER.info(f"Red Hat build of Kueue operator installed: {rh_kueue_operator_installed}")
    LOGGER.info(f"DSC Kueue management state: {dsc_kueue_state}")

    # Scenario 1: No Red Hat build of Kueue operator + DSC Kueue = Managed
    if not rh_kueue_operator_installed and dsc_kueue_state == DscComponents.ManagementState.MANAGED:
        LOGGER.info("Detected scenario: RHOAI managed Kueue")
        return KueueInstallationScenario.RHOAI_MANAGED

    # Scenario 2: Red Hat build of Kueue operator + DSC Kueue = Unmanaged
    elif rh_kueue_operator_installed and dsc_kueue_state == DscComponents.ManagementState.UNMANAGED:
        LOGGER.info("Detected scenario: Red Hat build of Kueue operator with RHOAI kueue component unmanaged")
        return KueueInstallationScenario.RHOAI_UNMANAGED

    else:
        LOGGER.warning(
            f"Unknown Kueue installation scenario: "
            f"Red Hat build of Kueue operator installed={rh_kueue_operator_installed}, "
            f"RHOAI kueue dsc_state={dsc_kueue_state}"
        )
        return KueueInstallationScenario.UNKNOWN


def should_patch_kueue_config(scenario: KueueInstallationScenario) -> bool:
    """
    Determine if kueue-manager-config should be patched based on installation scenario.

    Args:
        scenario: The detected Kueue installation scenario

    Returns:
        bool: True if config should be patched, False otherwise
    """
    if scenario == KueueInstallationScenario.RHOAI_MANAGED:
        return True
    elif scenario == KueueInstallationScenario.RHOAI_UNMANAGED:
        return False
    else:
        LOGGER.warning(f"Unknown scenario {scenario}, defaulting to not patching config")
        return False


def should_restart_kueue_deployment(scenario: KueueInstallationScenario) -> bool:
    """
    Determine if kueue-controller-manager deployment should be restarted based on installation scenario.

    Args:
        scenario: The detected Kueue installation scenario

    Returns:
        bool: True if deployment should be restarted, False otherwise
    """
    # Same logic as config patching for now
    return should_patch_kueue_config(scenario=scenario)
