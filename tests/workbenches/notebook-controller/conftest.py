from typing import Generator, Any, Dict

import pytest
from kubernetes.dynamic import DynamicClient
from _pytest.fixtures import FixtureRequest
from pytest_testconfig import config as py_config

from utilities.kueue_utils import (
    create_local_queue,
    create_cluster_queue,
    create_resource_flavor,
    LocalQueue,
    ClusterQueue,
    ResourceFlavor,
)
from ocp_resources.namespace import Namespace
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.resource import ResourceEditor
from utilities.constants import Labels, Annotations
from utilities.infra import create_ns
from utilities.kueue_detection import (
    detect_kueue_installation_scenario,
    should_patch_kueue_config,
    should_restart_kueue_deployment,
)
import yaml
import logging

LOGGER = logging.getLogger(__name__)


def kueue_resource_groups_for_notebooks(
    flavor_name: str,
    cpu_quota: str,
    memory_quota: str,
) -> list[Dict[str, Any]]:
    """Create resource groups configuration for Kueue with notebook-specific resources"""
    return [
        {
            "coveredResources": ["cpu", "memory"],
            "flavors": [
                {
                    "name": flavor_name,
                    "resources": [
                        {"name": "cpu", "nominalQuota": cpu_quota},
                        {"name": "memory", "nominalQuota": memory_quota},
                    ],
                }
            ],
        }
    ]


@pytest.fixture(scope="class")
def kueue_notebook_resource_flavor(
    request: FixtureRequest,
    admin_client: DynamicClient,
) -> Generator[ResourceFlavor, Any, Any]:
    """Create a ResourceFlavor for notebook workloads"""
    with create_resource_flavor(
        client=admin_client,
        name=request.param["name"],
        teardown=True,
    ) as resource_flavor:
        yield resource_flavor


@pytest.fixture(scope="class")
def kueue_notebook_cluster_queue(
    request: FixtureRequest,
    admin_client: DynamicClient,
) -> Generator[ClusterQueue, Any, Any]:
    """Create a ClusterQueue for notebook workloads"""
    resource_groups = kueue_resource_groups_for_notebooks(
        flavor_name=request.param["resource_flavor_name"],
        cpu_quota=request.param["cpu_quota"],
        memory_quota=request.param["memory_quota"],
    )

    with create_cluster_queue(
        client=admin_client,
        name=request.param["name"],
        resource_groups=resource_groups,
        namespace_selector=request.param.get("namespace_selector", {}),
        teardown=True,
    ) as cluster_queue:
        yield cluster_queue


@pytest.fixture(scope="class")
def kueue_notebook_local_queue(
    request: FixtureRequest,
    admin_client: DynamicClient,
    kueue_enabled_notebook_namespace: Namespace,
) -> Generator[LocalQueue, Any, Any]:
    """Create a LocalQueue for notebook workloads"""
    with create_local_queue(
        client=admin_client,
        name=request.param["name"],
        cluster_queue=request.param["cluster_queue"],
        namespace=kueue_enabled_notebook_namespace.name,
        teardown=True,
    ) as local_queue:
        yield local_queue


@pytest.fixture(scope="function")
def kueue_notebook_persistent_volume_claim(
    request: FixtureRequest,
    kueue_enabled_notebook_namespace: Namespace,
    unprivileged_client: DynamicClient,
) -> Generator[PersistentVolumeClaim, Any, Any]:
    """Create a PersistentVolumeClaim in the Kueue-enabled namespace"""
    with PersistentVolumeClaim(
        client=unprivileged_client,
        name=request.param["name"],
        namespace=kueue_enabled_notebook_namespace.name,
        label={Labels.OpenDataHub.DASHBOARD: "true"},
        accessmodes=PersistentVolumeClaim.AccessMode.RWO,
        size="10Gi",
        volume_mode=PersistentVolumeClaim.VolumeMode.FILE,
    ) as pvc:
        yield pvc


@pytest.fixture(scope="class")
def kueue_enabled_notebook_namespace(
    request: FixtureRequest,
    admin_client: DynamicClient,
    unprivileged_client: DynamicClient,
) -> Generator[Namespace, Any, Any]:
    """Create a namespace with Kueue label enabled for notebook workloads"""

    namespace_name = request.param["name"]
    add_kueue_label = request.param.get("add-kueue-label", True)

    with create_ns(
        admin_client=admin_client,
        name=namespace_name,
        unprivileged_client=unprivileged_client,
        add_kueue_label=add_kueue_label,
        pytest_request=request,
    ) as ns:
        yield ns


@pytest.fixture(scope="class")
def patched_kueue_manager_config(
    admin_client: DynamicClient,
    request: FixtureRequest,
) -> Generator[ConfigMap | None, Any, Any]:
    """Conditionally patch the kueue-manager-config ConfigMap based on Kueue installation scenario

    This fixture:
    1. Detects the Kueue installation scenario
    2. For RHOAI managed scenario: patches ConfigMap and restarts deployment
    3. For Red Hat build of Kueue operator scenario: skips patching (no config/deployment present)
    4. Yields the ConfigMap (or None if skipped) for tests
    5. On teardown: restores original config if it was patched
    """
    # Detect Kueue installation scenario
    scenario = detect_kueue_installation_scenario(client=admin_client)
    LOGGER.info(f"Detected Kueue installation scenario: {scenario}")

    # Check if we should patch config for this scenario
    if not should_patch_kueue_config(scenario):
        LOGGER.info(f"Skipping kueue-manager-config patching for scenario: {scenario}")
        yield None
        return

    namespace = py_config["applications_namespace"]
    config_map_name = "kueue-manager-config"

    # Get the existing ConfigMap
    try:
        config_map = ConfigMap(
            client=admin_client,
            name=config_map_name,
            namespace=namespace,
            ensure_exists=True,
        )
    except Exception as e:
        LOGGER.warning(f"Could not find kueue-manager-config ConfigMap: {e}")
        LOGGER.info("This is expected for Red Hat build of Kueue operator scenario")
        yield None
        return

    # Store original data and annotations for restoration
    original_data = config_map.instance.data.copy() if config_map.instance.data else {}
    original_annotations = (
        config_map.instance.metadata.annotations.copy() if config_map.instance.metadata.annotations else {}
    )

    LOGGER.info("Storing original kueue-manager-config data for restoration")

    # Get current config data
    current_data = config_map.instance.data or {}
    config_yaml = current_data.get("controller_manager_config.yaml", "{}")

    # Parse the existing configuration
    try:
        config_dict = yaml.safe_load(config_yaml) or {}
    except yaml.YAMLError:
        config_dict = {}

    # Ensure integrations section exists
    if "integrations" not in config_dict:
        config_dict["integrations"] = {}

    if "frameworks" not in config_dict["integrations"]:
        config_dict["integrations"]["frameworks"] = []

    # Add pod and statefulset if not already present
    frameworks = config_dict["integrations"]["frameworks"]
    if "pod" not in frameworks:
        frameworks.append("pod")
    if "statefulset" not in frameworks:
        frameworks.append("statefulset")

    # Convert back to YAML
    updated_config_yaml = yaml.dump(config_dict, default_flow_style=False)
    updated_data = {**current_data, "controller_manager_config.yaml": updated_config_yaml}

    # Apply the patch with both data and metadata annotations
    patch = {"metadata": {"annotations": {Annotations.OpenDataHubIo.MANAGED: "false"}}, "data": updated_data}

    def restart_kueue_deployment(reason: str):
        """Helper function to restart the kueue-controller-manager deployment"""
        if not should_restart_kueue_deployment(scenario):
            LOGGER.info(f"Skipping kueue-controller-manager deployment restart for scenario: {scenario}")
            return

        LOGGER.info(f"Restarting kueue-controller-manager deployment - {reason}")

        try:
            kueue_deployment = Deployment(
                client=admin_client,
                name="kueue-controller-manager",
                namespace=namespace,
                ensure_exists=True,
            )

            # Get current replica count before restart
            current_replicas = kueue_deployment.replicas
            if current_replicas is None:
                current_replicas = 1
            LOGGER.info(f"Current kueue-controller-manager replicas: {current_replicas}")

            # Restart deployment by scaling to 0 and back to original count
            LOGGER.info("Scaling kueue-controller-manager deployment to 0 replicas...")
            kueue_deployment.scale_replicas(replica_count=0)
            kueue_deployment.wait_for_replicas(deployed=False)
            LOGGER.info("kueue-controller-manager deployment scaled down to 0 replicas")

            # Now scale back up to original count
            LOGGER.info(f"Scaling kueue-controller-manager deployment back to {current_replicas} replicas...")
            kueue_deployment.scale_replicas(replica_count=current_replicas)
            kueue_deployment.wait_for_replicas(deployed=True)

            LOGGER.info(f"kueue-controller-manager deployment restart completed - {reason}")
        except Exception as e:
            LOGGER.warning(f"Could not restart kueue-controller-manager deployment: {e}")
            LOGGER.info("This is expected for Red Hat build of Kueue operator scenario")

    with ResourceEditor(patches={config_map: patch}):
        # After patching the ConfigMap, restart the deployment to pick up new configuration
        restart_kueue_deployment(reason="to apply patched configuration")
        yield config_map

    # Teardown: Restore original configuration and restart deployment
    LOGGER.info("Restoring original kueue-manager-config configuration")

    # Restore original data and annotations
    restore_patch = {"metadata": {"annotations": original_annotations}, "data": original_data}

    with ResourceEditor(patches={config_map: restore_patch}):
        # Restart deployment to pick up the restored original configuration
        restart_kueue_deployment(reason="to restore original configuration")
        LOGGER.info("Original kueue-manager-config configuration restored successfully")
