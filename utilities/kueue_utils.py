from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import pytest
import structlog
from kubernetes.client.exceptions import ApiException
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.api_service import APIService
from ocp_resources.cluster_service_version import ClusterServiceVersion
from ocp_resources.deployment import Deployment
from ocp_resources.job import Job
from ocp_resources.mutating_webhook_config import MutatingWebhookConfiguration
from ocp_resources.pod import Pod
from ocp_resources.resource import MissingRequiredArgumentError, NamespacedResource, Resource, ResourceEditor
from ocp_resources.validating_webhook_config import ValidatingWebhookConfiguration
from pytest_testconfig import config as py_config
from timeout_sampler import TimeoutExpiredError, TimeoutSampler, retry

from utilities.constants import Timeout
from utilities.resources.admission_check import AdmissionCheck

LOGGER = structlog.get_logger(name=__name__)

KUEUE_QUEUE_NAME_LABEL: str = "kueue.x-k8s.io/queue-name"
KUEUE_MANAGED_LABEL: str = "kueue.x-k8s.io/managed"
KUEUE_CLUSTER_QUEUE_LABEL: str = "kueue.x-k8s.io/cluster-queue-name"
KUEUE_LOCAL_QUEUE_LABEL: str = "kueue.x-k8s.io/local-queue-name"
KUEUE_OPERATOR_NAMESPACE: str = "openshift-kueue-operator"
KUEUE_CONTROLLER_LABEL_SELECTOR: str = "app.openshift.io/name=kueue"
KUEUE_VISIBILITY_API_GROUP: str = "visibility.kueue.x-k8s.io"  # gitleaks:allow
_KUEUE_WEBHOOK_NAMES: frozenset[str] = frozenset({
    "kueue-mutating-webhook-configuration",
    "kueue-validating-webhook-configuration",
})


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


def count_pods_started(labels: list[str], namespace: str, admin_client: DynamicClient) -> int:
    """Count pods that have started, i.e. reached Running or a terminal phase.

    A short-lived Job pod can pass through Running between two polls and be
    observed only as Succeeded. Callers that need evidence the pod was admitted
    and actually started should use this rather than counting Running pods, which
    misses fast-completing pods entirely.

    Args:
        labels: Label selectors to match pods on.
        namespace: Namespace to search.
        admin_client: Kubernetes client with admin privileges.

    Returns:
        The number of matching pods in Running, Succeeded or Failed phase.
    """
    pods = list(
        Pod.get(
            label_selector=",".join(labels),
            namespace=namespace,
            client=admin_client,
        )
    )
    return sum(1 for pod in pods if pod.instance.status.phase in ("Running", "Succeeded", "Failed"))


def wait_for_queue_active(queue: Resource, timeout: int = Timeout.TIMEOUT_2MIN) -> None:
    """Wait for a ClusterQueue or LocalQueue to report the Active condition.

    Queue creation can succeed while the Kueue controller is still starting
    (e.g. right after a fresh install or a config-change rollout). A queue
    only admits workloads once its Active condition is True, so waiting here
    prevents jobs submitted immediately after setup from being stuck pending.

    Args:
        queue: ClusterQueue or LocalQueue resource.
        timeout: Maximum seconds to wait for Active=True.

    Raises:
        TimeoutExpiredError: If the queue does not become Active in time.
    """

    def _is_active() -> bool:
        conditions = (queue.instance.status or {}).get("conditions", [])
        return any(condition.get("type") == "Active" and condition.get("status") == "True" for condition in conditions)

    try:
        for active in TimeoutSampler(wait_timeout=timeout, sleep=5, func=_is_active):
            if active:
                LOGGER.info(f"{queue.kind} {queue.name} is active")
                return
    except TimeoutExpiredError:
        raise TimeoutExpiredError(
            f"{queue.kind} {queue.name} did not become Active within {timeout}s: "
            f"{(queue.instance.status or {}).get('conditions')}"
        ) from None


def get_kueue_controller_pod_uids(client: DynamicClient) -> set[str]:
    """Return the UIDs of the current kueue-controller-manager pods."""
    return {
        pod.instance.metadata.uid
        for pod in Pod.get(
            label_selector=KUEUE_CONTROLLER_LABEL_SELECTOR,
            namespace=KUEUE_OPERATOR_NAMESPACE,
            client=client,
        )
    }


def wait_for_kueue_controller_rollout(
    client: DynamicClient,
    baseline_pod_uids: set[str],
    timeout: int = Timeout.TIMEOUT_4MIN,
) -> None:
    """Wait for the Kueue controller pods to be replaced after a config change.

    Changing the Kueue CR configuration (e.g. enabling a framework integration)
    makes the operator roll out new controller pods. The old pods stay Ready
    while they run the OLD configuration, so readiness checks alone return too
    early — a batch Job submitted in that window is ignored by Kueue and never
    gets a Workload. This waits until no pod from before the config change is
    left. If no rollout is observed within the timeout, raises
    TimeoutExpiredError with a descriptive failure message.

    Args:
        client: Kubernetes client with admin privileges.
        baseline_pod_uids: Controller pod UIDs captured before the config change.
        timeout: Maximum seconds to wait for the pod set to be replaced.

    Raises:
        ValueError: If ``baseline_pod_uids`` is empty.
        TimeoutExpiredError: If the controller pods are not replaced within the timeout.
    """
    if not baseline_pod_uids:
        raise ValueError(
            "baseline_pod_uids is empty; capture the controller pod UIDs before patching the "
            "Kueue CR, otherwise the first poll would falsely report a completed rollout."
        )

    try:
        for pod_uids in TimeoutSampler(
            wait_timeout=timeout,
            sleep=5,
            func=get_kueue_controller_pod_uids,
            client=client,
        ):
            if pod_uids and not pod_uids & baseline_pod_uids:
                LOGGER.info("Kueue controller rolled out with the new configuration")
                return
    except TimeoutExpiredError:
        raise TimeoutExpiredError(
            f"Kueue controller pods were not replaced within {timeout}s after the config change. "
            "Batch Jobs submitted now may be silently ignored by pods running the old configuration."
        ) from None


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


def _wait_for_kueue_controller_pods_gone(admin_client: DynamicClient, timeout: int = 60) -> None:
    """Wait for kueue-controller-manager pods to terminate, logging (not raising) on timeout."""
    try:
        for pods_gone in TimeoutSampler(
            wait_timeout=timeout,
            sleep=5,
            func=lambda: (
                not any(
                    Pod.get(
                        label_selector=KUEUE_CONTROLLER_LABEL_SELECTOR,
                        namespace=KUEUE_OPERATOR_NAMESPACE,
                        client=admin_client,
                    )
                )
            ),
        ):
            if pods_gone:
                break
    except TimeoutExpiredError:
        LOGGER.warning(f"Kueue controller pods did not terminate within {timeout}s, proceeding")


def pause_kueue_controller(admin_client: DynamicClient) -> int:
    """Scale the Kueue controller to 0 and remove visibility APIServices.

    Returns the original replica count so the caller can restore it after the
    operation that required the pause (typically a namespace deletion). The
    replica count is returned even if removing the visibility APIServices
    fails, so the caller can still resume the controller instead of leaving
    it paused indefinitely.
    """
    controller = Deployment(
        client=admin_client,
        name="kueue-controller-manager",
        namespace=KUEUE_OPERATOR_NAMESPACE,
    )
    if not controller.exists:
        LOGGER.warning("kueue-controller-manager deployment not found, skipping pause")
        return 0

    replicas = controller.instance.spec.replicas
    original_replicas = 1 if replicas is None else replicas
    LOGGER.info(f"Pausing kueue-controller-manager (scaling from {original_replicas} to 0)")
    ResourceEditor(patches={controller: {"spec": {"replicas": 0}}}).update()
    _wait_for_kueue_controller_pods_gone(admin_client=admin_client)

    try:
        remove_kueue_visibility_api_services(admin_client=admin_client)
    except ResourceNotFoundError, ApiException:
        LOGGER.warning(
            "Failed to remove Kueue visibility APIServices while pausing the controller; "
            "continuing so the caller can still resume it",
            exc_info=True,
        )
    return original_replicas


def resume_kueue_controller(admin_client: DynamicClient, replicas: int) -> None:
    """Scale the Kueue controller back to the given replica count."""
    if replicas <= 0:
        return
    controller = Deployment(
        client=admin_client,
        name="kueue-controller-manager",
        namespace=KUEUE_OPERATOR_NAMESPACE,
    )
    if not controller.exists:
        LOGGER.warning("kueue-controller-manager deployment not found, skipping resume")
        return
    LOGGER.info(f"Resuming kueue-controller-manager (scaling to {replicas})")
    ResourceEditor(patches={controller: {"spec": {"replicas": replicas}}}).update()


def remove_kueue_visibility_api_services(admin_client: DynamicClient, wait: bool = False) -> None:
    """Remove visibility APIServices that slow namespace deletion.

    Only removes APIServices — does NOT scale the controller or delete
    webhooks. The controller may recreate the APIServices, but namespace
    deletion still proceeds (just slower). Use ``full_kueue_controller_cleanup``
    when tearing down a Kueue install the caller owns.

    Args:
        admin_client: Kubernetes client with admin privileges.
        wait: If True, block until each APIService is actually deleted.
    """
    for api_service in APIService.get(client=admin_client):
        if api_service.name.endswith(KUEUE_VISIBILITY_API_GROUP):
            LOGGER.info(f"Removing Kueue visibility APIService {api_service.name}")
            api_service.delete(wait=wait)


def full_kueue_controller_cleanup(admin_client: DynamicClient) -> None:
    """Full teardown for a Kueue install owned by the caller.

    Scales the controller to 0, removes visibility APIServices, and deletes
    webhook configurations so a subsequent operator uninstall is clean.
    """
    controller = Deployment(
        client=admin_client,
        name="kueue-controller-manager",
        namespace=KUEUE_OPERATOR_NAMESPACE,
    )
    if controller.exists:
        LOGGER.info("Scaling kueue-controller-manager to 0 before cleanup")
        try:
            ResourceEditor(patches={controller: {"spec": {"replicas": 0}}}).update()
            _wait_for_kueue_controller_pods_gone(admin_client=admin_client)
        except ResourceNotFoundError, ApiException:
            LOGGER.warning("Failed to scale down kueue-controller-manager; continuing cleanup", exc_info=True)

    try:
        remove_kueue_visibility_api_services(admin_client=admin_client)
    except ResourceNotFoundError, ApiException:
        LOGGER.warning("Failed to remove Kueue visibility APIServices during cleanup", exc_info=True)

    for webhook in MutatingWebhookConfiguration.get(client=admin_client):
        if webhook.name in _KUEUE_WEBHOOK_NAMES:
            LOGGER.info(f"Removing MutatingWebhookConfiguration {webhook.name}")
            webhook.delete(wait=False)
    for webhook in ValidatingWebhookConfiguration.get(client=admin_client):
        if webhook.name in _KUEUE_WEBHOOK_NAMES:
            LOGGER.info(f"Removing ValidatingWebhookConfiguration {webhook.name}")
            webhook.delete(wait=False)


def drain_namespace_kueue_resources(admin_client: DynamicClient, namespace: str) -> None:
    """Delete all Jobs and Workloads in the namespace before namespace teardown.

    Kueue Workloads hold a `resource-in-use` finalizer on their ClusterQueue.
    If the namespace is deleted while Workloads still exist, the namespace
    gets stuck in Terminating and blocks ClusterQueue cleanup.
    """
    for job in Job.get(client=admin_client, namespace=namespace):
        LOGGER.info(f"Draining: deleting Job {job.name}")
        job.delete(wait=False)
    for workload in Workload.get(client=admin_client, namespace=namespace):
        LOGGER.info(f"Draining: deleting Workload {workload.name}")
        workload.delete(wait=False)

    def remaining_kueue_resources() -> list[str]:
        return [
            f"{resource.kind}/{resource.name}"
            for resource_cls in (Job, Workload)
            for resource in resource_cls.get(client=admin_client, namespace=namespace)
        ]

    remaining: list[str] = []
    try:
        for remaining in TimeoutSampler(
            wait_timeout=Timeout.TIMEOUT_4MIN,
            sleep=5,
            func=remaining_kueue_resources,
        ):
            if not remaining:
                LOGGER.info(f"Namespace {namespace} drained of Jobs and Workloads")
                return
    except TimeoutExpiredError:
        LOGGER.warning(f"Namespace {namespace} still holds Kueue resources after drain: {remaining}")
