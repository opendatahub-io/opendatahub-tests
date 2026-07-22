from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, Dict, Generator, List, Optional

from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.pod import Pod
from ocp_resources.resource import MissingRequiredArgumentError, NamespacedResource, Resource


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
        namespace_selector: Optional[Dict[str, Any]] = None,
        resource_groups: Optional[List[Dict[str, Any]]] = None,
        admission_checks: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        """
        Args:
            namespace_selector: Namespace selector to use
            resource_groups: Resource groups to use
            admission_checks: List of AdmissionCheck names to require on this queue
            kwargs: Keyword arguments to pass to the ClusterQueue constructor
        """
        super().__init__(
            **kwargs,
        )
        self.namespace_selector = namespace_selector
        self.resource_groups = resource_groups
        self.admission_checks = admission_checks

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
            if self.admission_checks:
                _spec["admissionChecksStrategy"] = {
                    "admissionChecks": [{"name": ac} for ac in self.admission_checks],
                }


class AdmissionCheck(Resource):
    api_group: str = "kueue.x-k8s.io"

    def __init__(
        self,
        controller_name: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.controller_name = controller_name

    def to_dict(self) -> None:
        super().to_dict()
        if not self.kind_dict and not self.yaml_file:
            if not self.controller_name:
                raise MissingRequiredArgumentError(argument="controller_name")
            self.res["spec"] = {"controllerName": self.controller_name}


class Workload(NamespacedResource):
    api_group: str = "kueue.x-k8s.io"


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
    admission_checks: Optional[List[str]] = None,
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
        admission_checks=admission_checks,
        teardown=teardown,
    ) as cluster_queue:
        yield cluster_queue


@contextmanager
def create_admission_check(
    client: DynamicClient,
    name: str,
    controller_name: str,
    teardown: bool = True,
) -> Generator[AdmissionCheck, Any, Any]:
    """
    Context manager to create and optionally delete an AdmissionCheck.
    """
    with AdmissionCheck(
        client=client,
        name=name,
        controller_name=controller_name,
        teardown=teardown,
    ) as admission_check:
        yield admission_check


def wait_for_deployments(labels: list[str], namespace: str, admin_client: DynamicClient) -> int:
    deployments = list(
        Deployment.get(
            label_selector=",".join(labels),
            namespace=namespace,
            dyn_client=admin_client,
        )
    )
    return len(deployments)


def get_workload_for_job(
    client: DynamicClient,
    job_uid: str,
    namespace: str,
) -> Optional[Workload]:
    """Find the Kueue Workload auto-created for a batch Job."""
    workloads = list(
        Workload.get(
            client=client,
            namespace=namespace,
            label_selector=f"kueue.x-k8s.io/job-uid={job_uid}",
        )
    )
    return workloads[0] if workloads else None


def check_workload_admitted(workload: Workload) -> bool:
    """Check if a Kueue Workload has Admitted=True condition."""
    conditions = getattr(workload.instance.status, "conditions", None) or []
    return any(
        (c.get("type") if isinstance(c, dict) else getattr(c, "type", None)) == "Admitted"
        and (c.get("status") if isinstance(c, dict) else getattr(c, "status", None)) == "True"
        for c in conditions
    )


def check_workload_quota_reserved(workload: Workload) -> bool:
    """Check if a Kueue Workload has QuotaReserved=True condition."""
    conditions = getattr(workload.instance.status, "conditions", None) or []
    return any(
        (c.get("type") if isinstance(c, dict) else getattr(c, "type", None)) == "QuotaReserved"
        and (c.get("status") if isinstance(c, dict) else getattr(c, "status", None)) == "True"
        for c in conditions
    )


def check_admission_check_active(admission_check: AdmissionCheck) -> bool:
    """Check if an AdmissionCheck has Active=True condition."""
    conditions = getattr(admission_check.instance.status, "conditions", None) or []
    return any(
        (c.get("type") if isinstance(c, dict) else getattr(c, "type", None)) == "Active"
        and (c.get("status") if isinstance(c, dict) else getattr(c, "status", None)) == "True"
        for c in conditions
    )


def check_cluster_queue_has_admission_check(cluster_queue: "ClusterQueue", admission_check_name: str) -> bool:
    """Check if a ClusterQueue still references an AdmissionCheck in its admissionChecksStrategy."""
    spec = cluster_queue.instance.spec
    strategy = getattr(spec, "admissionChecksStrategy", None)
    if not strategy:
        return False
    checks = getattr(strategy, "admissionChecks", None) or []
    return any(
        (c.get("name") if isinstance(c, dict) else getattr(c, "name", None)) == admission_check_name for c in checks
    )


def activate_admission_check(
    client: DynamicClient,
    admission_check_name: str,
) -> None:
    """Patch an AdmissionCheck's status to Active=True so the ClusterQueue can admit workloads.

    Acts as a fake AdmissionCheck Controller for upgrade testing. Uses a merge-patch
    to set Active=True with a synthetic reason, avoiding the need to deploy a real
    controller (e.g. ProvisioningRequest/MultiKueue).
    """
    api = client.resources.get(
        api_version=AdmissionCheck.api_group + "/v1beta1",
        kind="AdmissionCheck",
    )
    api.status.patch(
        name=admission_check_name,
        body={
            "status": {
                "conditions": [
                    {
                        "type": "Active",
                        "status": "True",
                        "reason": "FakeControllerReady",
                        "message": "Simulated controller for upgrade testing",
                        "lastTransitionTime": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    }
                ]
            }
        },
        content_type="application/merge-patch+json",
    )


def approve_admission_check_on_workload(
    client: DynamicClient,
    workload: Workload,
    admission_check_name: str,
) -> None:
    """Patch a Workload's status to set an AdmissionCheck state to Ready.

    Uses JSON merge-patch, which replaces ``status.admissionChecks`` entirely.
    Safe while there is exactly one AdmissionCheck and no reliance on sibling
    fields like ``podSetUpdates``. Callers with multiple checks should
    read-modify-write instead.
    """
    api = client.resources.get(
        api_version=Workload.api_group + "/v1beta1",
        kind="Workload",
    )
    api.status.patch(
        name=workload.name,
        namespace=workload.namespace,
        body={
            "status": {
                "admissionChecks": [
                    {
                        "name": admission_check_name,
                        "state": "Ready",
                        "message": "Approved by upgrade test",
                        "lastTransitionTime": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    }
                ]
            }
        },
        content_type="application/merge-patch+json",
    )


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
