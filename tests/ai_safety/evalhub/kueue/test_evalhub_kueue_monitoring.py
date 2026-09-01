import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.job import Job
from ocp_resources.namespace import Namespace
from ocp_resources.resource import ResourceEditor
from ocp_resources.service import Service
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_safety.evalhub.utils import (
    WORKLOAD_INADMISSIBLE_REASONS,
    cleanup_evalhub_job,
    cluster_queue_name,
    evalhub_runtime_label_selector,
    log_job_kueue_labels,
    submit_evalhub_kueue_job_and_get_id,
    wait_for_evalhub_job_workload_inadmissible,
)
from utilities.constants import Timeout
from utilities.kueue_utils import ClusterQueue, LocalQueue, get_workload_condition

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.kueue
@pytest.mark.tier2
class TestEvalHubKueueMonitoring:
    """Monitoring/troubleshooting signals for gated EvalHub jobs.

    Rather than depend on quota sizing or cluster capacity to hold a job pending,
    both tests create a deterministic gated state by stopping the ClusterQueue
    (``stopPolicy: HoldAndDrain``) -- the same mechanism the queue-management
    tests use -- and then assert on the monitoring signals a stuck job exposes.
    """

    def test_gated_job_batch_job_reports_suspended(
        self,
        admin_client: DynamicClient,
        evalhub_kueue_namespace: Namespace,
        evalhub_kueue_single_job_local_queue: LocalQueue,
        evalhub_kueue_vllm_service: Service,
        evalhub_kueue_request_common: dict[str, str],
    ) -> None:
        """TC-SUSPEND-001: A gated EvalHub job's Kubernetes Job reports Suspended=True.

        Kueue admits a workload by clearing ``spec.suspend`` on its batch Job; a
        workload that cannot be admitted keeps its Job suspended. The blog points
        operators at exactly this signal when a job is not making progress, so we
        assert the gated job's Job carries ``spec.suspend: true`` and a
        ``Suspended=True`` status condition.
        """
        common = evalhub_kueue_request_common
        namespace = evalhub_kueue_namespace.name
        cluster_queue = ClusterQueue(client=admin_client, name=cluster_queue_name(evalhub_kueue_single_job_local_queue))
        job_id = None

        try:
            with ResourceEditor(patches={cluster_queue: {"spec": {"stopPolicy": "HoldAndDrain"}}}):
                job_id = submit_evalhub_kueue_job_and_get_id(
                    request_common=common,
                    local_queue_name=evalhub_kueue_single_job_local_queue.name,
                    model_service_name=evalhub_kueue_vllm_service.name,
                    tenant_namespace=namespace,
                    job_name="tc-suspend-001-gated",
                )

                try:
                    wait_for_evalhub_job_workload_inadmissible(
                        admin_client=admin_client,
                        namespace=namespace,
                        evalhub_job_id=job_id,
                        timeout=Timeout.TIMEOUT_10MIN,
                    )
                except TimeoutExpiredError:
                    log_job_kueue_labels(admin_client, namespace, job_id)
                    raise

                selector = evalhub_runtime_label_selector(evalhub_job_id=job_id)
                gated_jobs = list(Job.get(client=admin_client, namespace=namespace, label_selector=selector))
                assert gated_jobs, f"No Kubernetes Job found for gated EvalHub job {job_id}"
                gated_job = gated_jobs[0]

                suspend = (gated_job.instance.spec or {}).get("suspend")
                assert suspend is True, f"Expected gated Job spec.suspend=True, got {suspend!r}"

                # The Suspended status condition may trail spec.suspend by a beat.
                try:
                    for job_sample in TimeoutSampler(
                        wait_timeout=Timeout.TIMEOUT_2MIN,
                        sleep=5,
                        func=lambda: Job(client=admin_client, name=gated_job.name, namespace=namespace),
                    ):
                        conditions = (job_sample.instance.status or {}).get("conditions", [])
                        if any(c.get("type") == "Suspended" and c.get("status") == "True" for c in conditions):
                            break
                except TimeoutExpiredError:
                    pytest.fail(f"Gated Job {gated_job.name} never reported Suspended=True condition")
        finally:
            if job_id:
                try:
                    cleanup_evalhub_job(**common, job_id=job_id)
                except Exception:
                    LOGGER.warning(f"Failed to clean up job {job_id}", exc_info=True)

    def test_gated_workload_reports_inadmissible_message(
        self,
        admin_client: DynamicClient,
        evalhub_kueue_namespace: Namespace,
        evalhub_kueue_single_job_local_queue: LocalQueue,
        evalhub_kueue_vllm_service: Service,
        evalhub_kueue_request_common: dict[str, str],
    ) -> None:
        """TC-QUOTA-MSG-001: A gated Workload carries a human-readable diagnostic message.

        The blog tells operators to run ``kubectl describe workload`` and read the
        condition message to learn *why* a job is stuck. We assert the gated
        Workload's ``QuotaReserved`` condition is ``False`` with an inadmissible
        reason and a non-empty message, so this troubleshooting step yields a
        useful answer rather than an empty/opaque status.

        Note: the deterministic gate here is a stopped ClusterQueue
        (``stopPolicy: HoldAndDrain``), which reports an inadmissible/suspended
        reason without depending on quota sizing or cluster capacity.
        """
        common = evalhub_kueue_request_common
        namespace = evalhub_kueue_namespace.name
        cluster_queue = ClusterQueue(client=admin_client, name=cluster_queue_name(evalhub_kueue_single_job_local_queue))
        job_id = None

        try:
            with ResourceEditor(patches={cluster_queue: {"spec": {"stopPolicy": "HoldAndDrain"}}}):
                job_id = submit_evalhub_kueue_job_and_get_id(
                    request_common=common,
                    local_queue_name=evalhub_kueue_single_job_local_queue.name,
                    model_service_name=evalhub_kueue_vllm_service.name,
                    tenant_namespace=namespace,
                    job_name="tc-quota-msg-001-gated",
                )

                try:
                    gated_workload = wait_for_evalhub_job_workload_inadmissible(
                        admin_client=admin_client,
                        namespace=namespace,
                        evalhub_job_id=job_id,
                        timeout=Timeout.TIMEOUT_10MIN,
                    )
                except TimeoutExpiredError:
                    log_job_kueue_labels(admin_client, namespace, job_id)
                    raise

                condition = get_workload_condition(workload=gated_workload, condition_type="QuotaReserved")
                assert condition is not None, (
                    f"Gated workload {gated_workload.name} has no QuotaReserved condition: "
                    f"{(gated_workload.instance.status or {}).get('conditions', [])}"
                )
                assert condition.get("status") == "False", (
                    f"Expected QuotaReserved=False on gated workload, got {condition.get('status')!r}"
                )
                assert condition.get("reason") in WORKLOAD_INADMISSIBLE_REASONS, (
                    f"Expected QuotaReserved.reason in {WORKLOAD_INADMISSIBLE_REASONS}, got {condition.get('reason')!r}"
                )
                assert condition.get("message"), (
                    "Expected a non-empty QuotaReserved message to aid troubleshooting, got "
                    f"{condition.get('message')!r}"
                )
        finally:
            if job_id:
                try:
                    cleanup_evalhub_job(**common, job_id=job_id)
                except Exception:
                    LOGGER.warning(f"Failed to clean up job {job_id}", exc_info=True)
