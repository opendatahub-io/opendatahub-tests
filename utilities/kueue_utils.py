from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.pod import Pod

from utilities.resources.cluster_queue import ClusterQueue as ClusterQueue
from utilities.resources.kueue import Kueue as Kueue  # noqa: F401
from utilities.resources.local_queue import LocalQueue as LocalQueue
from utilities.resources.resource_flavor import ResourceFlavor as ResourceFlavor
from utilities.resources.workload import Workload as Workload  # noqa: F401

KUEUE_QUEUE_NAME_LABEL: str = "kueue.x-k8s.io/queue-name"
KUEUE_MANAGED_LABEL: str = "kueue.x-k8s.io/managed"


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
    resource_groups: List[Dict[str, Any]],
    namespace_selector: Optional[Dict[str, Any]] = None,
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
