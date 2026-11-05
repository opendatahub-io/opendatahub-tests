import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.resource import ResourceEditor
from ocp_resources.service import Service
from timeout_sampler import TimeoutExpiredError

from tests.ai_safety.evalhub.utils import (
    WORKLOAD_INADMISSIBLE_REASONS,
    build_evalhub_kueue_job_payload,
    cleanup_evalhub_job,
    cluster_queue_name,
    log_job_kueue_labels,
    submit_evalhub_job,
    validate_evalhub_job_completed,
    wait_for_evalhub_job,
    wait_for_evalhub_job_workload_inadmissible,
)
from utilities.constants import Timeout
from utilities.kueue_utils import ClusterQueue, LocalQueue

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.kueue
@pytest.mark.tier2
class TestEvalHubKueueQueueRecovery:
    """Verify the queue accepts new jobs after being stopped and re-enabled."""

    def test_new_job_succeeds_after_queue_reenabled(
        self,
        admin_client: DynamicClient,
        evalhub_kueue_namespace: Namespace,
        evalhub_kueue_single_job_local_queue: LocalQueue,
        evalhub_kueue_vllm_service: Service,
        evalhub_kueue_request_common: dict[str, str],
    ) -> None:
        """TC-QM-001: A gated job resumes and completes after a stopped ClusterQueue is re-enabled.

        Kueue does not discard a workload that is held while its ClusterQueue is
        stopped (`HoldAndDrain`); the workload stays pending until the queue is
        re-enabled, at which point Kueue admits it and the job runs to
        completion. EvalHub does not proactively fail inadmissible jobs. This
        test verifies both properties: the originally-gated job recovers and
        completes once the queue is restored, and a freshly submitted job also
        completes.
        """
        common = evalhub_kueue_request_common

        cluster_queue = ClusterQueue(client=admin_client, name=cluster_queue_name(evalhub_kueue_single_job_local_queue))
        job_ids: list[str] = []

        try:
            with ResourceEditor(patches={cluster_queue: {"spec": {"stopPolicy": "HoldAndDrain"}}}):
                data = submit_evalhub_job(
                    **common,
                    payload=build_evalhub_kueue_job_payload(
                        queue_name=evalhub_kueue_single_job_local_queue.name,
                        model_service_name=evalhub_kueue_vllm_service.name,
                        tenant_namespace=evalhub_kueue_namespace.name,
                        job_name="tc-qm-001-gated",
                    ),
                )
                gated_job_id = data["resource"]["id"]
                job_ids.append(gated_job_id)

                try:
                    wait_for_evalhub_job_workload_inadmissible(
                        admin_client=admin_client,
                        namespace=evalhub_kueue_namespace.name,
                        evalhub_job_id=gated_job_id,
                        timeout=Timeout.TIMEOUT_10MIN,
                    )
                except TimeoutExpiredError:
                    log_job_kueue_labels(admin_client, evalhub_kueue_namespace.name, gated_job_id)
                    raise

            gated_result = wait_for_evalhub_job(**common, job_id=gated_job_id, timeout=Timeout.TIMEOUT_10MIN)
            validate_evalhub_job_completed(job_data=gated_result)

            data = submit_evalhub_job(
                **common,
                payload=build_evalhub_kueue_job_payload(
                    queue_name=evalhub_kueue_single_job_local_queue.name,
                    model_service_name=evalhub_kueue_vllm_service.name,
                    tenant_namespace=evalhub_kueue_namespace.name,
                    job_name="tc-qm-001-recovery",
                ),
            )
            recovery_job_id = data["resource"]["id"]
            job_ids.append(recovery_job_id)

            job_result = wait_for_evalhub_job(**common, job_id=recovery_job_id, timeout=Timeout.TIMEOUT_10MIN)
            validate_evalhub_job_completed(job_data=job_result)
        finally:
            for jid in job_ids:
                try:
                    cleanup_evalhub_job(**common, job_id=jid)
                except Exception:
                    LOGGER.warning(f"Failed to clean up job {jid}", exc_info=True)


@pytest.mark.kueue
@pytest.mark.tier2
class TestEvalHubKueueWorkloadConditions:
    """Verify Kueue workload conditions expose useful diagnostic info when a job is gated."""

    def test_workload_quota_conditions_when_queue_full(
        self,
        admin_client: DynamicClient,
        evalhub_kueue_namespace: Namespace,
        evalhub_kueue_single_job_local_queue: LocalQueue,
        evalhub_kueue_vllm_service: Service,
        evalhub_kueue_request_common: dict[str, str],
    ) -> None:
        """TC-QM-002: A gated workload carries QuotaReserved=False/Inadmissible with a message.

        Operators troubleshooting a stuck-pending job should be able to inspect
        the workload's QuotaReserved condition and get a meaningful error message.

        Uses HoldAndDrain to create a deterministic inadmissible state for the job,
        then inspects the workload conditions without relying on quota-exhaustion timing.
        """
        common = evalhub_kueue_request_common

        cluster_queue = ClusterQueue(client=admin_client, name=cluster_queue_name(evalhub_kueue_single_job_local_queue))
        job_id = None

        try:
            with ResourceEditor(patches={cluster_queue: {"spec": {"stopPolicy": "HoldAndDrain"}}}):
                data = submit_evalhub_job(
                    **common,
                    payload=build_evalhub_kueue_job_payload(
                        queue_name=evalhub_kueue_single_job_local_queue.name,
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
                        timeout=Timeout.TIMEOUT_10MIN,
                    )
                except TimeoutExpiredError:
                    log_job_kueue_labels(admin_client, evalhub_kueue_namespace.name, job_id)
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
                assert quota_reserved.get("reason") in WORKLOAD_INADMISSIBLE_REASONS, (
                    f"Expected QuotaReserved.reason in {WORKLOAD_INADMISSIBLE_REASONS}, got: {quota_reserved}"
                )
                assert quota_reserved.get("message"), (
                    "Expected a non-empty QuotaReserved message to aid troubleshooting"
                )

            try:
                wait_for_evalhub_job(**common, job_id=job_id, timeout=60)
            except TimeoutExpiredError:
                LOGGER.warning(
                    f"Job {job_id} did not reach a terminal state within 60s after the "
                    "ClusterQueue hold was lifted; ignoring since the test's assertions already passed"
                )
        finally:
            if job_id:
                try:
                    cleanup_evalhub_job(**common, job_id=job_id)
                except Exception:
                    LOGGER.warning(f"Failed to clean up job {job_id}", exc_info=True)
