from contextlib import contextmanager
from typing import Any, Generator, Callable
from kubernetes.dynamic import DynamicClient
from ocp_resources.resource import NamespacedResource, Resource, MissingRequiredArgumentError
from ocp_resources.deployment import Deployment
from ocp_resources.pod import Pod
from ocp_resources.config_map import ConfigMap
import yaml
from simple_logger.logger import get_logger

LOGGER = get_logger(name=__name__)


class ResourceFlavor(Resource):
    api_group: str = "kueue.x-k8s.io"

    def __init__(self, **kwargs: Any):
        """
        Args:
            kwargs: Keyword arguments to pass to the ResourceFlavor constructor
        """
        super().__init__(
            **kwargs,
        )

    def to_dict(self) -> None:
        super().to_dict()
        if not self.kind_dict and not self.yaml_file:
            self.res["spec"] = {}


class LocalQueue(NamespacedResource):
    api_group: str = "kueue.x-k8s.io"

    def __init__(
        self,
        cluster_queue: str,
        **kwargs: Any,
    ):
        """
        Args:
            cluster_queue: Name of the cluster queue to use
            kwargs: Keyword arguments to pass to the LocalQueue constructor
        """
        super().__init__(
            **kwargs,
        )
        self.cluster_queue = cluster_queue

    def to_dict(self) -> None:
        super().to_dict()
        if not self.kind_dict and not self.yaml_file:
            if not self.cluster_queue:
                raise MissingRequiredArgumentError(argument="cluster_queue")
            self.res["spec"] = {}
            _spec = self.res["spec"]
            _spec["clusterQueue"] = self.cluster_queue


class ClusterQueue(Resource):
    api_group: str = "kueue.x-k8s.io"

    def __init__(
        self,
        namespace_selector: dict[str, Any] | None = None,
        resource_groups: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            namespace_selector: Namespace selector to use
            resource_groups: Resource groups to use
            kwargs: Keyword arguments to pass to the ClusterQueue constructor
        """
        super().__init__(
            **kwargs,
        )
        self.namespace_selector = namespace_selector
        self.resource_groups = resource_groups

    def to_dict(self) -> None:
        super().to_dict()
        if not self.kind_dict and not self.yaml_file:
            if not self.resource_groups:
                raise MissingRequiredArgumentError(argument="resource_groups")
            self.res["spec"] = {}
            _spec = self.res["spec"]
            if self.namespace_selector is not None:
                _spec["namespaceSelector"] = self.namespace_selector
            else:
                _spec["namespaceSelector"] = {}
            if self.resource_groups:
                _spec["resourceGroups"] = self.resource_groups


@contextmanager
def create_resource_flavor(
    client: DynamicClient,
    name: str,
    teardown: bool = True,
) -> Generator[ResourceFlavor, Any, Any]:
    """
    Context manager to create and optionally delete a ResourceFlavor.
    """
    with ResourceFlavor(
        client=client,
        name=name,
        teardown=teardown,
    ) as resource_flavor:
        yield resource_flavor


@contextmanager
def create_local_queue(
    client: DynamicClient,
    name: str,
    cluster_queue: str,
    namespace: str,
    teardown: bool = True,
) -> Generator[LocalQueue, Any, Any]:
    """
    Context manager to create and optionally delete a LocalQueue.
    """
    with LocalQueue(
        client=client,
        name=name,
        cluster_queue=cluster_queue,
        namespace=namespace,
        teardown=teardown,
    ) as local_queue:
        yield local_queue


@contextmanager
def create_cluster_queue(
    client: DynamicClient,
    name: str,
    resource_groups: list[dict[str, Any]],
    namespace_selector: dict[str, Any] | None = None,
    teardown: bool = True,
) -> Generator[ClusterQueue, Any, Any]:
    """
    Context manager to create and optionally delete a ClusterQueue.
    """
    with ClusterQueue(
        client=client,
        name=name,
        resource_groups=resource_groups,
        namespace_selector=namespace_selector,
        teardown=teardown,
    ) as cluster_queue:
        yield cluster_queue


def wait_for_deployments(labels: list[str], namespace: str, admin_client: DynamicClient) -> int:
    deployments = list(
        Deployment.get(
            label_selector=",".join(labels),
            namespace=namespace,
            dyn_client=admin_client,
        )
    )
    return len(deployments)


def check_gated_pods_and_running_pods(
    labels: list[str], namespace: str, admin_client: DynamicClient
) -> tuple[int, int]:
    running_pods = 0
    gated_pods = 0
    pods = list(
        Pod.get(
            label_selector=",".join(labels),
            namespace=namespace,
            dyn_client=admin_client,
        )
    )
    for pod in pods:
        if pod.instance.status.phase == "Running":
            running_pods += 1
        elif pod.instance.status.phase == "Pending":
            if all(
                condition.type == "PodScheduled"
                and condition.status == "False"
                and condition.reason == "SchedulingGated"
                for condition in pod.instance.status.conditions
            ):
                gated_pods += 1
    return running_pods, gated_pods


def get_queue_resource_usage(queue: ClusterQueue | LocalQueue, flavor_name: str) -> dict[str, str | bool]:
    """
    Get resource usage for a specific flavor in a queue.

    Args:
        queue: ClusterQueue or LocalQueue object
        flavor_name: Name of the resource flavor to check

    Returns:
        Dict containing CPU and memory usage information
    """
    usage_info: dict[str, str | bool] = {"cpu": "0", "memory": "0", "found": False}

    # Refresh the queue status
    queue.get()

    if hasattr(queue.instance, "status") and queue.instance.status:
        flavors_usage = queue.instance.status.get("flavorsUsage", [])

        for flavor_usage in flavors_usage:
            if flavor_usage.get("name") == flavor_name:
                usage_info["found"] = True
                resources = flavor_usage.get("resources", [])

                for resource in resources:
                    resource_name = resource.get("name")
                    total_usage = resource.get("total", "0")

                    if resource_name in ["cpu", "memory"] and isinstance(total_usage, str):
                        usage_info[resource_name] = str(total_usage)
                break

    return usage_info


def verify_queue_tracks_workload(
    queue: ClusterQueue | LocalQueue,
    flavor_name: str,
    expected_cpu: str | None = None,
    expected_memory: str | None = None,
) -> bool:
    """
    Verify that a queue is tracking workload resource usage.

    The function has two modes of operation:
    - When expected_cpu and/or expected_memory are provided: Returns True only if
      ALL provided expected values match exactly (AND logic)
    - When no expected values are provided: Returns True if either CPU or memory
      usage is non-zero (OR logic)

    Args:
        queue: ClusterQueue or LocalQueue object
        flavor_name: Name of the resource flavor to check
        expected_cpu: Expected CPU usage (optional)
        expected_memory: Expected memory usage (optional)

    Returns:
        True if the queue is tracking the workload according to the mode, False otherwise
    """
    usage_info = get_queue_resource_usage(queue=queue, flavor_name=flavor_name)

    if not usage_info["found"]:
        return False

    # If specific expected values are provided, check for exact match
    if expected_cpu is not None or expected_memory is not None:
        cpu_matches = True
        memory_matches = True

        if expected_cpu is not None:
            cpu_matches = usage_info["cpu"] == expected_cpu

        if expected_memory is not None:
            memory_matches = usage_info["memory"] == expected_memory

        # When checking specific values, both must match (AND logic)
        return cpu_matches and memory_matches
    else:
        # Default behavior: check if there's any resource usage (OR logic)
        has_cpu_usage = usage_info["cpu"] != "0"
        has_memory_usage = usage_info["memory"] != "0"
        return has_cpu_usage or has_memory_usage


def backup_configmap_data(config_map: ConfigMap) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Backup ConfigMap data and annotations for later restoration.

    Args:
        config_map: ConfigMap object to backup

    Returns:
        Tuple of (original_data, original_annotations)
    """
    configmap_instance = config_map.instance

    original_data = configmap_instance.data.copy() if configmap_instance.data else {}
    original_annotations = (
        configmap_instance.metadata.annotations.copy() if configmap_instance.metadata.annotations else {}
    )

    LOGGER.info("Stored original ConfigMap data and annotations for restoration")
    return original_data, original_annotations


def update_kueue_config_frameworks(config_yaml: str, frameworks_to_add: list[str]) -> str:
    """
    Update the Kueue configuration to add specified frameworks to the integrations section.

    Args:
        config_yaml: Current configuration YAML string
        frameworks_to_add: List of frameworks to add

    Returns:
        Updated configuration YAML string

    Raises:
        yaml.YAMLError: If the given configuration is malformed
    """
    # Parse the existing configuration - fail fast if corrupted
    try:
        config_dict = yaml.safe_load(config_yaml) or {}
    except yaml.YAMLError as e:
        LOGGER.error(f"Failed to parse Kueue configuration - this indicates a serious issue: {e}")
        LOGGER.error(f"Corrupted YAML content: {config_yaml}")
        raise

    # Validate expected structure exists
    if "integrations" not in config_dict:
        LOGGER.error(f"Kueue configuration missing 'integrations' section: {config_dict}")
        raise ValueError(
            "Kueue configuration is missing required 'integrations' section. "
            "This indicates a malformed configuration that needs investigation."
        )

    if "frameworks" not in config_dict["integrations"]:
        LOGGER.error(f"Kueue configuration missing 'frameworks' section in integrations: {config_dict['integrations']}")
        raise ValueError(
            "Kueue configuration is missing required 'frameworks' section in integrations. "
            "This indicates a malformed configuration that needs investigation."
        )

    # Add frameworks if not already present
    frameworks = config_dict["integrations"]["frameworks"]
    for framework in frameworks_to_add:
        if framework not in frameworks:
            frameworks.append(framework)
            LOGGER.info(f"Added '{framework}' framework to Kueue configuration")

    # Convert back to YAML
    return yaml.dump(config_dict, default_flow_style=False)


def restart_kueue_deployment(
    client: DynamicClient,
    namespace: str,
    reason: str,
    should_restart_func: Callable[[str], bool],
    scenario: str,
    deployment_name: str = "kueue-controller-manager",
) -> None:
    """
    Restart the kueue-controller-manager deployment.

    Args:
        client: Kubernetes dynamic client
        namespace: Namespace where the deployment is located
        reason: Reason for the restart (for logging)
        should_restart_func: Function to check if restart should be performed
        scenario: Kueue installation scenario
        deployment_name: Name of the deployment (default: "kueue-controller-manager")
    """
    if not should_restart_func(scenario):
        LOGGER.info(f"Skipping {deployment_name} deployment restart for scenario: {scenario}")
        return

    LOGGER.info(f"Restarting {deployment_name} deployment - {reason}")

    try:
        kueue_deployment = Deployment(
            client=client,
            name=deployment_name,
            namespace=namespace,
            ensure_exists=True,
        )

        # Get current replica count before restart
        current_replicas = kueue_deployment.replicas
        if current_replicas is None:
            current_replicas = 1
        LOGGER.info(f"Current {deployment_name} replicas: {current_replicas}")

        # Restart deployment by scaling to 0 and back to original count
        LOGGER.info(f"Scaling {deployment_name} deployment to 0 replicas...")
        kueue_deployment.scale_replicas(replica_count=0)
        kueue_deployment.wait_for_replicas(deployed=False)
        LOGGER.info(f"{deployment_name} deployment scaled down to 0 replicas")

        # Now scale back up to original count
        LOGGER.info(f"Scaling {deployment_name} deployment back to {current_replicas} replicas...")
        kueue_deployment.scale_replicas(replica_count=current_replicas)
        kueue_deployment.wait_for_replicas(deployed=True)

        LOGGER.info(f"{deployment_name} deployment restart completed - {reason}")
    except Exception as e:
        LOGGER.warning(f"Could not restart {deployment_name} deployment: {e}")
        LOGGER.info("This is expected for Red Hat build of Kueue operator scenario")


def create_kueue_config_patch(
    current_data: dict[str, Any], updated_config_yaml: str, managed_annotation: str = "false"
) -> dict[str, Any]:
    """
    Create a patch for the Kueue ConfigMap with updated configuration and annotations.

    Args:
        current_data: Current ConfigMap data
        updated_config_yaml: Updated configuration YAML string
        managed_annotation: Value for the managed annotation (default: "false")

    Returns:
        Patch dictionary for the ConfigMap
    """
    from utilities.constants import Annotations

    updated_data = {**current_data, "controller_manager_config.yaml": updated_config_yaml}

    return {"metadata": {"annotations": {Annotations.OpenDataHubIo.MANAGED: managed_annotation}}, "data": updated_data}
