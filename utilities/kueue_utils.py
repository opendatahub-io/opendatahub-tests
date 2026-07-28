from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.cluster_service_version import ClusterServiceVersion
from ocp_resources.pod import Pod
from ocp_resources.resource import MissingRequiredArgumentError, NamespacedResource, Resource
from pytest_testconfig import config as py_config
from timeout_sampler import TimeoutExpiredError, TimeoutSampler, retry

from utilities.constants import Timeout
from utilities.resources.admission_check import AdmissionCheck

LOGGER = structlog.get_logger(name=__name__)

KUEUE_QUEUE_NAME_LABEL: str = "kueue.x-k8s.io/queue-name"
KUEUE_MANAGED_LABEL: str = "kueue.x-k8s.io/managed"
KUEUE_CLUSTER_QUEUE_LABEL: str = "kueue.x-k8s.io/cluster-queue-name"
KUEUE_LOCAL_QUEUE_LABEL: str = "kueue.x-k8s.io/local-queue-name"


def is_kueue_operator_installed(admin_client: DynamicClient) -> bool:
    """Return True if a succeeded Kueue operator CSV is present."""
    try:
        csvs = list(
            ClusterServiceVersion.get(
                client=admin_client,
                namespace=py_config.get("applications_namespace", "openshift-operators"),
            )
        )
        for csv in csvs:
            if csv.name.startswith("kueue") and csv.status == csv.Status.SUCCEEDED:
                LOGGER.info(f"Found Kueue operator CSV: {csv.name}")
                return True
        return False
    except ResourceNotFoundError:
        return False


class ResourceFlavor(Resource):
    """Kueue ResourceFlavor resource."""

    api_group: str = "kueue.x-k8s.io"
    api_version: str = "kueue.x-k8s.io/v1beta2"

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
    """Kueue LocalQueue resource."""

    api_group: str = "kueue.x-k8s.io"
    api_version: str = "kueue.x-k8s.io/v1beta2"

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
    """Kueue ClusterQueue resource."""

    api_group: str = "kueue.x-k8s.io"
    api_version: str = "kueue.x-k8s.io/v1beta2"

    def __init__(
        self,
        namespace_selector: dict[str, Any] | None = None,
        resource_groups: list[dict[str, Any]] | None = None,
        admission_checks: list[str] | None = None,
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


class Workload(NamespacedResource):
    """Kueue Workload resource (kueue.x-k8s.io/v1beta2)."""

    api_group: str = "kueue.x-k8s.io"
    api_version: str = "kueue.x-k8s.io/v1beta2"


class Kueue(Resource):
    """Kueue CR of the Red Hat build of Kueue operator (kueue.openshift.io/v1)."""

    api_group: str = "kueue.openshift.io"
    api_version: str = "kueue.openshift.io/v1"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        management_state: str | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            config: Kueue controller configuration (e.g. framework integrations)
            management_state: managementState for the Kueue controller
            kwargs: Keyword arguments to pass to the Kueue constructor
        """
        super().__init__(
            **kwargs,
        )
        self.config = config
        self.management_state = management_state

    def to_dict(self) -> None:
        super().to_dict()
        if not self.kind_dict and not self.yaml_file:
            self.res["spec"] = {}
            _spec = self.res["spec"]
            if self.config is not None:
                _spec["config"] = self.config
            if self.management_state is not None:
                _spec["managementState"] = self.management_state


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


@contextmanager
def create_cluster_queue(
    client: DynamicClient,
    name: str,
    resource_groups: list[dict[str, Any]],
    namespace_selector: dict[str, Any] | None = None,
    admission_checks: list[str] | None = None,
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


def get_workload_for_job(
    client: DynamicClient,
    job_uid: str,
    namespace: str,
) -> Workload | None:
    """Find the Kueue Workload auto-created for a batch Job."""
    workloads = list(
        Workload.get(
            client=client,
            namespace=namespace,
            label_selector=f"kueue.x-k8s.io/job-uid={job_uid}",
        )
    )
    if len(workloads) > 1:
        raise ValueError(f"Multiple Workloads ({len(workloads)}) found for Job UID {job_uid}")
    return workloads[0] if workloads else None


def wait_for_workload_condition(
    client: DynamicClient,
    workload_name: str,
    namespace: str,
    condition_check: Callable[[Workload], bool],
    condition_name: str,
    timeout: int = Timeout.TIMEOUT_2MIN,
) -> None:
    """Poll a Workload until a condition is met, or fail the test."""
    try:
        for workload in TimeoutSampler(
            wait_timeout=timeout,
            sleep=5,
            func=lambda: Workload(
                client=client,
                name=workload_name,
                namespace=namespace,
            ),
        ):
            if workload.exists and condition_check(workload):
                return
    except TimeoutExpiredError:
        pytest.fail(f"Workload '{workload_name}' did not reach {condition_name}")


def check_workload_admitted(workload: Workload) -> bool:
    """Check if a Kueue Workload has Admitted=True condition."""
    conditions = getattr(workload.instance.status, "conditions", None) or []
    return any(
        (condition.get("type") if isinstance(condition, dict) else getattr(condition, "type", None)) == "Admitted"
        and (condition.get("status") if isinstance(condition, dict) else getattr(condition, "status", None)) == "True"
        for condition in conditions
    )


def check_workload_quota_reserved(workload: Workload) -> bool:
    """Check if a Kueue Workload has QuotaReserved=True condition."""
    conditions = getattr(workload.instance.status, "conditions", None) or []
    return any(
        (condition.get("type") if isinstance(condition, dict) else getattr(condition, "type", None)) == "QuotaReserved"
        and (condition.get("status") if isinstance(condition, dict) else getattr(condition, "status", None)) == "True"
        for condition in conditions
    )


def check_admission_check_active(admission_check: AdmissionCheck) -> bool:
    """Check if an AdmissionCheck has Active=True condition."""
    conditions = getattr(admission_check.instance.status, "conditions", None) or []
    return any(
        (condition.get("type") if isinstance(condition, dict) else getattr(condition, "type", None)) == "Active"
        and (condition.get("status") if isinstance(condition, dict) else getattr(condition, "status", None)) == "True"
        for condition in conditions
    )


def check_cluster_queue_has_admission_check(cluster_queue: ClusterQueue, admission_check_name: str) -> bool:
    """Check if a ClusterQueue still references an AdmissionCheck in its admissionChecksStrategy."""
    spec = cluster_queue.instance.spec
    strategy = getattr(spec, "admissionChecksStrategy", None)
    if not strategy:
        return False
    checks = getattr(strategy, "admissionChecks", None) or []
    return any(
        (check.get("name") if isinstance(check, dict) else getattr(check, "name", None)) == admission_check_name
        for check in checks
    )


def activate_admission_check(
    client: DynamicClient,
    admission_check_name: str,
) -> None:
    """Patch an AdmissionCheck's status to Active=True so the ClusterQueue can admit workloads.

    Acts as a fake AdmissionCheck Controller for upgrade testing. Uses a merge-patch
    to set Active=True with a synthetic reason, avoiding the need to deploy a real
    controller (e.g. ProvisioningRequest/MultiKueue).

    Uses ``api.status.patch()`` because this targets the Kubernetes ``/status``
    subresource endpoint.  ``ResourceEditor`` and ``Resource.update()`` only patch
    the main resource endpoint; the API server silently ignores status fields there.
    """
    ac = AdmissionCheck(client=client, name=admission_check_name)
    ac.api.status.patch(
        name=ac.name,
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
    workload: Workload,
    admission_check_name: str,
) -> None:
    """Patch a Workload's status to set an AdmissionCheck state to Ready.

    Uses JSON merge-patch, which replaces ``status.admissionChecks`` entirely.
    Safe while there is exactly one AdmissionCheck and no reliance on sibling
    fields like ``podSetUpdates``. Callers with multiple checks should
    read-modify-write instead.

    Uses ``api.status.patch()`` because this targets the Kubernetes ``/status``
    subresource endpoint.  ``ResourceEditor`` and ``Resource.update()`` only patch
    the main resource endpoint; the API server silently ignores status fields there.
    """
    workload.api.status.patch(
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
    pods = list(
        Pod.get(
            label_selector="app.openshift.io/name=kueue",
            namespace="openshift-kueue-operator",
            client=client,
        )
    )
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
