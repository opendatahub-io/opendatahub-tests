"""Queue management tests for EvalHub Kueue integration.

Covers admission control and diagnostic scenarios:
- TC-QM-001: Two sequential jobs complete when quota allows only one at a time.
- TC-QM-002: Workload conditions surface actionable diagnostic info for operators.

## Quota and timing design

The `evalhub_kueue_single_job_local_queue` ClusterQueue quota
(SINGLE_JOB_CPU_QUOTA / SINGLE_JOB_MEMORY_QUOTA) is sized to hold exactly one
EvalHub evaluation job (~300m CPU, ~640Mi memory). This ensures a second job cannot
be admitted while the first holds the quota.

TC-QM-001 submits jobs sequentially (job1 then job2) and asserts both complete.
EvalHub processes jobs asynchronously so job2's Kubernetes batch Job may not exist
until after job1 completes — the test focuses on the end-to-end completion assertion.

TC-QM-002 uses `stopPolicy: HoldAndDrain` on the ClusterQueue to create a
deterministic inadmissible state without relying on quota-exhaustion timing.
"""

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.job import Job
from ocp_resources.namespace import Namespace
from ocp_resources.resource import ResourceEditor
from ocp_resources.route import Route
from ocp_resources.service import Service
from timeout_sampler import TimeoutExpiredError

from tests.ai_safety.evalhub.utils import (
    build_evalhub_job_payload,
    evalhub_runtime_label_selector,
    submit_evalhub_job,
    wait_for_evalhub_job,
    wait_for_evalhub_job_workload_admitted,
    wait_for_evalhub_job_workload_inadmissible,
)
from utilities.kueue_utils import ClusterQueue, LocalQueue

LOGGER = structlog.get_logger(name=__name__)

KUEUE_QUEUE_LABEL = "kueue.x-k8s.io/queue-name"


def _log_job_kueue_labels(admin_client: DynamicClient, namespace: str, evalhub_job_id: str) -> None:
    """Log the Kueue label on the Kubernetes Job created by EvalHub.

    Kueue reads the queue-name from a Job LABEL (not annotation).
    Helps diagnose whether EvalHub is propagating the queue-name label —
    if the label is missing, Kueue will never create a Workload.
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


def _kueue_payload(local_queue: LocalQueue, **kwargs) -> dict:
    """Build a job payload with the Kueue queue field set."""
    payload = build_evalhub_job_payload(**kwargs)
    payload["queue"] = {"kind": "kueue", "name": local_queue.name}
    return payload


def _cluster_queue_name(local_queue: LocalQueue) -> str:
    """Return the ClusterQueue name backing this LocalQueue."""
    return local_queue.instance.spec.clusterQueue


@pytest.mark.kueue
@pytest.mark.tier2
class TestEvalHubKueueAutoResume:
    """Verify sequential jobs complete under a quota limited to one job at a time."""

    def test_job_auto_resume_after_quota_frees(
        self,
        admin_client: DynamicClient,
        evalhub_kueue_namespace: Namespace,
        evalhub_kueue_single_job_local_queue: LocalQueue,
        evalhub_kueue_user_token: str,
        evalhub_kueue_vllm_service: Service,
        evalhub_kueue_route: Route,
        evalhub_kueue_ca_bundle_file: str,
    ) -> None:
        """TC-QM-001: Both jobs complete when quota allows only one at a time.

        Validates the auto-resume scenario: job2 is submitted while job1 holds
        the quota. Kueue queues job2 until job1 finishes and quota frees, then
        admits and runs job2 automatically without any manual intervention.

        Note: EvalHub processes jobs asynchronously — job2's Kubernetes batch Job
        may not be created until after job1 completes (at which point Kueue admits
        it immediately). The core assertion is that both jobs reach `completed` state
        regardless of whether job2 was ever observed in the `Inadmissible` Kueue state.
        """
        common = {
            "host": evalhub_kueue_route.host,
            "token": evalhub_kueue_user_token,
            "ca_bundle_file": evalhub_kueue_ca_bundle_file,
            "tenant": evalhub_kueue_namespace.name,
        }

        # Job1 — submitted first to consume the single-job quota
        data1 = submit_evalhub_job(
            **common,
            payload=_kueue_payload(
                evalhub_kueue_single_job_local_queue,
                model_service_name=evalhub_kueue_vllm_service.name,
                tenant_namespace=evalhub_kueue_namespace.name,
                job_name="tc-qm-001-job1",
            ),
        )
        job1_id = data1["resource"]["id"]
        try:
            wait_for_evalhub_job_workload_admitted(
                admin_client=admin_client,
                namespace=evalhub_kueue_namespace.name,
                evalhub_job_id=job1_id,
            )
        except TimeoutExpiredError:
            _log_job_kueue_labels(admin_client, evalhub_kueue_namespace.name, job1_id)
            raise

        # Job2 — submitted while job1 holds the quota
        data2 = submit_evalhub_job(
            **common,
            payload=_kueue_payload(
                evalhub_kueue_single_job_local_queue,
                model_service_name=evalhub_kueue_vllm_service.name,
                tenant_namespace=evalhub_kueue_namespace.name,
                job_name="tc-qm-001-job2",
            ),
        )
        job2_id = data2["resource"]["id"]

        # Wait for job1 to finish — this releases quota for job2
        job1_result = wait_for_evalhub_job(**common, job_id=job1_id, timeout=600)
        assert job1_result["status"]["state"] == "completed", f"Job1 did not complete: {job1_result['status']}"

        # Job2 should run to completion once job1's quota is released.
        # Kueue's auto-admission ensures no manual intervention is needed.
        job2_result = wait_for_evalhub_job(**common, job_id=job2_id, timeout=600)
        assert job2_result["status"]["state"] == "completed", (
            f"Job2 did not complete after quota freed by job1: {job2_result['status']}"
        )


@pytest.mark.kueue
@pytest.mark.tier2
class TestEvalHubKueueWorkloadConditions:
    """Verify Kueue workload conditions expose useful diagnostic info when a job is gated."""

    def test_workload_quota_conditions_when_queue_full(
        self,
        admin_client: DynamicClient,
        evalhub_kueue_namespace: Namespace,
        evalhub_kueue_single_job_local_queue: LocalQueue,
        evalhub_kueue_user_token: str,
        evalhub_kueue_vllm_service: Service,
        evalhub_kueue_route: Route,
        evalhub_kueue_ca_bundle_file: str,
    ) -> None:
        """TC-QM-002: A gated workload carries QuotaReserved=False/Inadmissible with a message.

        Operators troubleshooting a stuck-pending job should be able to inspect
        the workload's QuotaReserved condition and get a meaningful error message.

        Uses HoldAndDrain to create a deterministic inadmissible state for the job,
        then inspects the workload conditions without relying on quota-exhaustion timing.
        """
        common = {
            "host": evalhub_kueue_route.host,
            "token": evalhub_kueue_user_token,
            "ca_bundle_file": evalhub_kueue_ca_bundle_file,
            "tenant": evalhub_kueue_namespace.name,
        }

        cq = ClusterQueue(client=admin_client, name=_cluster_queue_name(evalhub_kueue_single_job_local_queue))

        # Stop the queue so the submitted job's workload is reliably inadmissible
        # while we inspect its conditions.
        with ResourceEditor(patches={cq: {"spec": {"stopPolicy": "HoldAndDrain"}}}):
            data = submit_evalhub_job(
                **common,
                payload=_kueue_payload(
                    evalhub_kueue_single_job_local_queue,
                    model_service_name=evalhub_kueue_vllm_service.name,
                    tenant_namespace=evalhub_kueue_namespace.name,
                    job_name="tc-qm-002-job",
                ),
            )
            job_id = data["resource"]["id"]

            try:
                workload = wait_for_evalhub_job_workload_inadmissible(
                    admin_client=admin_client,
                    namespace=evalhub_kueue_namespace.name,
                    evalhub_job_id=job_id,
                )
            except TimeoutExpiredError:
                _log_job_kueue_labels(admin_client, evalhub_kueue_namespace.name, job_id)
                raise

            conditions = (workload.instance.status or {}).get("conditions", [])
            quota_reserved = next(
                (c for c in conditions if c.get("type") == "QuotaReserved"),
                None,
            )

            assert quota_reserved is not None, (
                f"Expected 'QuotaReserved' condition on gated workload, got: {conditions}"
            )
            assert quota_reserved["status"] == "False", (
                f"Expected QuotaReserved.status=False when job is inadmissible, got: {quota_reserved}"
            )
            assert quota_reserved.get("reason") == "Inadmissible", (
                f"Expected QuotaReserved.reason=Inadmissible, got: {quota_reserved}"
            )
            assert quota_reserved.get("message"), "Expected a non-empty QuotaReserved message to aid troubleshooting"

        # Queue is restored on context exit — the job now runs to completion
        wait_for_evalhub_job(**common, job_id=job_id, timeout=600)
