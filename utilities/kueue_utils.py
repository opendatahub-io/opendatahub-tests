from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.exceptions import MissingRequiredArgumentError
from ocp_resources.pod import Pod
from timeout_sampler import retry

from utilities.constants import Timeout
from utilities.resources.cluster_queue import ClusterQueue as _ClusterQueue
from utilities.resources.local_queue import LocalQueue as _LocalQueue
from utilities.resources.resource_flavor import ResourceFlavor
from utilities.resources.workload import Workload as _Workload
from utilities.resources.workload_priority_class import WorkloadPriorityClass

LOGGER = structlog.get_logger(name=__name__)


class Workload(_Workload):
    """Kueue Workload with helpers for status conditions."""

    def get_condition(self, condition_type: str) -> dict[str, Any] | None:
        """Return the status condition matching condition_type, or None."""
        self.get()
        for condition in self.instance.get("status", {}).get("conditions", []):
            if condition.get("type") == condition_type:
                return dict(condition)
        return None

    def is_admitted(self) -> bool:
        """Return True if Admitted condition is True."""
        condition = self.get_condition(condition_type="Admitted")
        return condition is not None and condition.get("status") == "True"


class LocalQueue(_LocalQueue):
    """LocalQueue that requires cluster_queue when building manifest data."""

    def to_dict(self) -> None:
        if not self.kind_dict and not self.yaml_file and not self.cluster_queue:
            raise MissingRequiredArgumentError(argument="cluster_queue")
        super().to_dict()


class ClusterQueue(_ClusterQueue):
    """ClusterQueue that requires resource_groups and defaults namespaceSelector."""

    def to_dict(self) -> None:
        if not self.kind_dict and not self.yaml_file and not self.resource_groups:
            raise MissingRequiredArgumentError(argument="resource_groups")
        super().to_dict()
        if not self.kind_dict and not self.yaml_file:
            spec = self.res.get("spec")
            if isinstance(spec, dict) and "namespaceSelector" not in spec:
                spec["namespaceSelector"] = self.namespace_selector if self.namespace_selector is not None else {}


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


def check_gated_pods_and_running_pods(
    labels: list[str], namespace: str, admin_client: DynamicClient
) -> tuple[int, int]:
    running_pods = 0
    gated_pods = 0
    pods = list(
        Pod.get(
            label_selector=",".join(labels),
            namespace=namespace,
            client=admin_client,
        )
    )
    for pod in pods:
        if pod.instance.status.phase == "Running":
            running_pods += 1
        elif pod.instance.status.phase == "Pending" and all(
            condition.type == "PodScheduled" and condition.status == "False" and condition.reason == "SchedulingGated"
            for condition in pod.instance.status.conditions
        ):
            gated_pods += 1
    return running_pods, gated_pods


@contextmanager
def create_workload_priority_class(
    client: DynamicClient,
    name: str,
    value: int,
    teardown: bool = True,
) -> Generator[WorkloadPriorityClass, Any, Any]:
    """Context manager to create and optionally delete a WorkloadPriorityClass."""
    with WorkloadPriorityClass(
        client=client,
        name=name,
        value=value,
        teardown=teardown,
    ) as wpc:
        yield wpc


def get_workloads_for_job(client: DynamicClient, namespace: str, job_name: str) -> list[Workload]:
    """Return Workload resources owned by the given Job name."""
    workloads = list(Workload.get(client=client, namespace=namespace))
    workload_instances = [wl.instance for wl in workloads]
    return [
        wl
        for wl, instance in zip(workloads, workload_instances, strict=False)
        if any(
            ref.get("name") == job_name and ref.get("kind") == "Job"
            for ref in instance.get("metadata", {}).get("ownerReferences", [])
        )
    ]


@retry(
    wait_timeout=Timeout.TIMEOUT_4MIN,
    sleep=5,
)
def wait_for_kueue_crds_available(client: DynamicClient) -> bool:
    """Wait for Kueue CRDs and controller to be fully available.

    This function waits for:
    1. Kueue CRDs to be registered in the API server
    2. kueue-controller-manager pods to be Ready (needed for webhooks/admission control)

    Raises:
        TimeoutExpiredError: If CRDs or controller are not available within the timeout period.

    Returns:
        True when CRDs are available and controller is ready.
    """
    # Check if CRDs are registered (raises exception if not, then will @retry)
    list(ResourceFlavor.get(client=client))

    # Check kueue-controller-manager pods exist and are ready
    # Kueue can be deployed in different namespaces (openshift-operators, openshift-kueue-operator, kueue-system)
    # Try to find pods in common locations
    pods = []
    for ns in ["openshift-operators", "openshift-kueue-operator", "kueue-system"]:
        ns_pods = list(
            Pod.get(
                label_selector="control-plane=controller-manager,app.kubernetes.io/name=kueue",
                namespace=ns,
                client=client,
            )
        )
        pods.extend(ns_pods)
    all_pods_ready = pods and all(
        any(
            condition.type == Pod.Condition.READY and condition.status == Pod.Condition.Status.TRUE
            for condition in pod.instance.status.conditions or []
        )
        for pod in pods
    )
    if not all_pods_ready:
        LOGGER.info("Kueue controller pods not ready yet, retrying...")
        return False

    LOGGER.info(f"Kueue is ready: CRDs available and {len(pods)} controller pod(s) running")
    return True
