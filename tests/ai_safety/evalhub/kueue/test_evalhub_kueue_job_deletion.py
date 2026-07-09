"""Job deletion tests for EvalHub Kueue integration.

Verifies that Kueue Workload objects are properly cleaned up when the backing
Kubernetes Job is deleted — both while gated (pending quota) and while actively
running. Orphaned workloads would incorrectly consume quota and block future
submissions.

Note: These tests delete the Kubernetes Job via the admin client rather than
the EvalHub API. The operator-generated kube-rbac-proxy auth.yaml only covers
the collection-level path (/api/v1/evaluations/jobs) and not individual job
paths (/api/v1/evaluations/jobs/{id}), which causes 403 on per-job DELETE
requests. This is a known operator bug tracked separately. Testing K8s-level
job deletion directly still validates the core Kueue integration: that Kueue
Workloads are cleaned up when their parent Job is removed.
"""

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.job import Job
from ocp_resources.namespace import Namespace
from ocp_resources.resource import ResourceEditor
from ocp_resources.route import Route
from ocp_resources.service import Service
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_safety.evalhub.utils import (
    _get_evalhub_job_workload,
    build_evalhub_job_payload,
    evalhub_runtime_label_selector,
    submit_evalhub_job,
    wait_for_evalhub_job_workload_admitted,
    wait_for_evalhub_job_workload_inadmissible,
)
from utilities.kueue_utils import ClusterQueue, LocalQueue, check_gated_pods_and_running_pods

LOGGER = structlog.get_logger(name=__name__)

KUEUE_QUEUE_LABEL = "kueue.x-k8s.io/queue-name"


def _log_job_kueue_labels(admin_client: DynamicClient, namespace: str, evalhub_job_id: str) -> None:
    """Log the Kueue labels on the Kubernetes Job created by EvalHub.

    Debugging helper — can be removed once Kueue label propagation is stable.
    """
    selector = evalhub_runtime_label_selector(evalhub_job_id=evalhub_job_id)
    jobs = list(Job.get(client=admin_client, namespace=namespace, label_selector=selector))
    if not jobs:
        LOGGER.warning("No Kubernetes Job found for EvalHub job", evalhub_job_id=evalhub_job_id)
        return
    for job in jobs:
        labels = job.instance.metadata.labels or {}
        queue_label = labels.get(KUEUE_QUEUE_LABEL)
        LOGGER.info(
            "Kubernetes Job kueue label check",
            job_name=job.name,
            kueue_queue_name_label=queue_label,
            has_kueue_label=queue_label is not None,
            all_labels=dict(labels),
        )


def _delete_k8s_job(admin_client: DynamicClient, namespace: str, evalhub_job_id: str) -> None:
    """Delete the Kubernetes batch Job for a given EvalHub job ID.

    Uses the admin client to delete the Job directly, bypassing the EvalHub
    API. This is required because the operator-managed kube-rbac-proxy
    auth.yaml lacks rules for individual job paths.
    """
    selector = evalhub_runtime_label_selector(evalhub_job_id=evalhub_job_id)
    jobs = list(Job.get(client=admin_client, namespace=namespace, label_selector=selector))
    if not jobs:
        LOGGER.warning("No Kubernetes Job found to delete", evalhub_job_id=evalhub_job_id)
        return
    for job in jobs:
        LOGGER.info(f"Deleting Kubernetes Job {job.name} for EvalHub job {evalhub_job_id}")
        job.delete(wait=True)
    LOGGER.info(f"Kubernetes Job(s) for EvalHub job {evalhub_job_id} deleted")


def _wait_for_workload_absent(
    admin_client: DynamicClient,
    namespace: str,
    evalhub_job_id: str,
    timeout: int = 60,
    sleep: int = 5,
) -> None:
    """Poll until the Kueue Workload for the given EvalHub job no longer exists."""
    try:
        for workload in TimeoutSampler(
            wait_timeout=timeout,
            sleep=sleep,
            func=_get_evalhub_job_workload,
            admin_client=admin_client,
            namespace=namespace,
            evalhub_job_id=evalhub_job_id,
        ):
            if workload is None:
                return
    except TimeoutExpiredError:
        raise TimeoutExpiredError(f"Kueue Workload for job {evalhub_job_id} still present after {timeout}s") from None


def _kueue_payload(local_queue: LocalQueue, **kwargs) -> dict:
    """Build a job payload with the Kueue queue field set."""
    payload = build_evalhub_job_payload(**kwargs)
    payload["queue"] = {"kind": "kueue", "name": local_queue.name}
    return payload


@pytest.mark.kueue
@pytest.mark.tier2
class TestEvalHubKueueJobDeletion:
    """Verify Kueue Workloads are cleaned up when EvalHub jobs are deleted."""

    def test_delete_pending_job_cleans_workload(
        self,
        admin_client: DynamicClient,
        evalhub_kueue_namespace: Namespace,
        evalhub_kueue_single_job_local_queue: LocalQueue,
        evalhub_kueue_user_token: str,
        evalhub_kueue_vllm_service: Service,
        evalhub_kueue_route: Route,
        evalhub_kueue_ca_bundle_file: str,
    ) -> None:
        """TC-DEL-001: Deleting a pending job's K8s Job removes its Kueue Workload.

        When a Kubernetes Job is deleted while the Workload is pending admission
        (ClusterQueue stopped), Kueue must remove the Workload to prevent quota
        leakage that would block future submissions.

        Uses `stopPolicy: HoldAndDrain` on the ClusterQueue to create a
        deterministic pending state without relying on quota-exhaustion timing
        (the vLLM emulator completes jobs too fast for quota-based gating).

        Note: Uses admin-level K8s Job deletion because the EvalHub API's
        per-job DELETE path is not covered by the operator-generated auth.yaml.
        """
        common = {
            "host": evalhub_kueue_route.host,
            "token": evalhub_kueue_user_token,
            "ca_bundle_file": evalhub_kueue_ca_bundle_file,
            "tenant": evalhub_kueue_namespace.name,
        }

        # Stop the ClusterQueue so all submitted workloads stay inadmissible.
        # This is more deterministic than quota exhaustion (vLLM emulator completes jobs
        # in seconds, making quota-timing tests flaky).
        cluster_queue = evalhub_kueue_single_job_local_queue.instance.spec.clusterQueue
        cq = ClusterQueue(client=admin_client, name=cluster_queue)
        with ResourceEditor(patches={cq: {"spec": {"stopPolicy": "HoldAndDrain"}}}):
            # Submit job while queue is stopped → Workload created but not admitted
            data = submit_evalhub_job(
                **common,
                payload=_kueue_payload(
                    evalhub_kueue_single_job_local_queue,
                    model_service_name=evalhub_kueue_vllm_service.name,
                    tenant_namespace=evalhub_kueue_namespace.name,
                    job_name="tc-del-001",
                ),
            )
            job_id = data["resource"]["id"]

            try:
                wait_for_evalhub_job_workload_inadmissible(
                    admin_client=admin_client,
                    namespace=evalhub_kueue_namespace.name,
                    evalhub_job_id=job_id,
                    timeout=300,
                )
            except TimeoutExpiredError:
                _log_job_kueue_labels(admin_client, evalhub_kueue_namespace.name, job_id)
                raise

            # Delete the pending job's K8s Job directly (admin operation)
            _delete_k8s_job(admin_client=admin_client, namespace=evalhub_kueue_namespace.name, evalhub_job_id=job_id)

        # Workload must be gone — no orphaned objects holding quota
        _wait_for_workload_absent(
            admin_client=admin_client,
            namespace=evalhub_kueue_namespace.name,
            evalhub_job_id=job_id,
        )

    def test_delete_running_job_cleans_workload(
        self,
        admin_client: DynamicClient,
        evalhub_kueue_namespace: Namespace,
        evalhub_kueue_multi_job_local_queue: LocalQueue,
        evalhub_kueue_user_token: str,
        evalhub_kueue_vllm_service: Service,
        evalhub_kueue_route: Route,
        evalhub_kueue_ca_bundle_file: str,
    ) -> None:
        """TC-DEL-002: Deleting an admitted (running) job's K8s Job removes its Workload.

        Kueue must release the reserved quota when a job's K8s Job object is
        deleted mid-execution, allowing other workloads to be admitted.

        Note: Uses admin-level K8s Job deletion because the EvalHub API's
        per-job DELETE path is not covered by the operator-generated auth.yaml.
        """
        common = {
            "host": evalhub_kueue_route.host,
            "token": evalhub_kueue_user_token,
            "ca_bundle_file": evalhub_kueue_ca_bundle_file,
            "tenant": evalhub_kueue_namespace.name,
        }

        data = submit_evalhub_job(
            **common,
            payload=_kueue_payload(
                evalhub_kueue_multi_job_local_queue,
                model_service_name=evalhub_kueue_vllm_service.name,
                tenant_namespace=evalhub_kueue_namespace.name,
                job_name="tc-del-002",
            ),
        )
        job_id = data["resource"]["id"]

        # Wait until admitted and pod is actively running
        try:
            wait_for_evalhub_job_workload_admitted(
                admin_client=admin_client,
                namespace=evalhub_kueue_namespace.name,
                evalhub_job_id=job_id,
            )
        except TimeoutExpiredError:
            _log_job_kueue_labels(admin_client, evalhub_kueue_namespace.name, job_id)
            raise

        selector = evalhub_runtime_label_selector(evalhub_job_id=job_id)
        try:
            for running, _ in TimeoutSampler(
                wait_timeout=300,
                sleep=5,
                func=check_gated_pods_and_running_pods,
                labels=[selector],
                namespace=evalhub_kueue_namespace.name,
                admin_client=admin_client,
            ):
                if running >= 1:
                    break
        except TimeoutExpiredError:
            pytest.fail(f"Pod for job {job_id} did not reach running state within 300s")

        # Delete the K8s Job directly while it is running (admin operation)
        _delete_k8s_job(admin_client=admin_client, namespace=evalhub_kueue_namespace.name, evalhub_job_id=job_id)

        # Workload should be removed, freeing quota for future submissions
        _wait_for_workload_absent(
            admin_client=admin_client,
            namespace=evalhub_kueue_namespace.name,
            evalhub_job_id=job_id,
        )
