import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.job import Job
from ocp_resources.namespace import Namespace
from ocp_resources.service import Service
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_safety.evalhub.utils import (
    EVALHUB_JOB_TERMINAL_STATES,
    build_evalhub_kueue_job_payload,
    check_workload_quota_reserved,
    cleanup_evalhub_job,
    evalhub_runtime_label_selector,
    get_job_status,
    submit_evalhub_job,
    validate_evalhub_job_completed,
    wait_for_evalhub_job,
    wait_for_evalhub_job_workload_admitted,
)
from utilities.constants import Timeout
from utilities.kueue_utils import LocalQueue, count_pods_started

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.kueue
@pytest.mark.tier1
class TestEvalHubKueueBasic:
    """Basic lifecycle tests for EvalHub jobs with Kueue admission control."""

    def test_evalhub_job_workload_created(
        self,
        admin_client: DynamicClient,
        evalhub_kueue_namespace: Namespace,
        evalhub_kueue_multi_job_local_queue: LocalQueue,
        evalhub_kueue_vllm_service: Service,
        evalhub_kueue_request_common: dict[str, str],
    ) -> None:
        """Submit an EvalHub job and verify Kueue Workload is created."""
        common = evalhub_kueue_request_common
        job_id = None

        try:
            data = submit_evalhub_job(
                **common,
                payload=build_evalhub_kueue_job_payload(
                    queue_name=evalhub_kueue_multi_job_local_queue.name,
                    model_service_name=evalhub_kueue_vllm_service.name,
                    tenant_namespace=evalhub_kueue_namespace.name,
                    job_name="tc-basic-001-workload",
                ),
            )
            job_id = data["resource"]["id"]

            workload = wait_for_evalhub_job_workload_admitted(
                admin_client=admin_client,
                namespace=evalhub_kueue_namespace.name,
                evalhub_job_id=job_id,
                timeout=Timeout.TIMEOUT_10MIN,
            )

            assert workload is not None
            assert check_workload_quota_reserved(workload)
        finally:
            if job_id:
                cleanup_evalhub_job(**common, job_id=job_id)

    def test_evalhub_job_lifecycle_with_kueue(
        self,
        admin_client: DynamicClient,
        evalhub_kueue_namespace: Namespace,
        evalhub_kueue_multi_job_local_queue: LocalQueue,
        evalhub_kueue_vllm_service: Service,
        evalhub_kueue_request_common: dict[str, str],
    ) -> None:
        """Full lifecycle: submit → admitted → running → completed."""
        common = evalhub_kueue_request_common
        job_id = None

        try:
            data = submit_evalhub_job(
                **common,
                payload=build_evalhub_kueue_job_payload(
                    queue_name=evalhub_kueue_multi_job_local_queue.name,
                    model_service_name=evalhub_kueue_vllm_service.name,
                    tenant_namespace=evalhub_kueue_namespace.name,
                    job_name="tc-basic-002-lifecycle",
                ),
            )
            job_id = data["resource"]["id"]

            wait_for_evalhub_job_workload_admitted(
                admin_client=admin_client,
                namespace=evalhub_kueue_namespace.name,
                evalhub_job_id=job_id,
                timeout=Timeout.TIMEOUT_10MIN,
            )

            selector = evalhub_runtime_label_selector(evalhub_job_id=job_id)
            try:
                for started_pods in TimeoutSampler(
                    wait_timeout=Timeout.TIMEOUT_10MIN,
                    sleep=5,
                    func=count_pods_started,
                    labels=[selector],
                    namespace=evalhub_kueue_namespace.name,
                    admin_client=admin_client,
                ):
                    if started_pods >= 1:
                        break
            except TimeoutExpiredError:
                pytest.fail(f"Pod for admitted job {job_id} never started")

            job_result = wait_for_evalhub_job(**common, job_id=job_id, timeout=Timeout.TIMEOUT_10MIN)
            validate_evalhub_job_completed(job_data=job_result)

            jobs = list(
                Job.get(
                    client=admin_client,
                    namespace=evalhub_kueue_namespace.name,
                    label_selector=selector,
                )
            )
            assert len(jobs) >= 1
            jobs[0].wait_for_condition(condition="Complete", status="True", timeout=Timeout.TIMEOUT_2MIN)
        finally:
            if job_id:
                cleanup_evalhub_job(**common, job_id=job_id)

    def test_evalhub_status_reflects_kueue_state(
        self,
        admin_client: DynamicClient,
        evalhub_kueue_namespace: Namespace,
        evalhub_kueue_multi_job_local_queue: LocalQueue,
        evalhub_kueue_vllm_service: Service,
        evalhub_kueue_request_common: dict[str, str],
    ) -> None:
        """EvalHub API status should transition: pending → running → completed.

        The emulator can finish a job between two polls, so missing the
        transient ``running`` state is tolerated — the stable assertion is
        that the job starts as ``pending`` and ends as ``completed``.
        """
        common = evalhub_kueue_request_common
        job_id = None

        try:
            data = submit_evalhub_job(
                **common,
                payload=build_evalhub_kueue_job_payload(
                    queue_name=evalhub_kueue_multi_job_local_queue.name,
                    model_service_name=evalhub_kueue_vllm_service.name,
                    tenant_namespace=evalhub_kueue_namespace.name,
                    job_name="tc-basic-003-status",
                ),
            )
            job_id = data["resource"]["id"]

            assert data["status"]["state"] == "pending"

            try:
                for sample in TimeoutSampler(
                    wait_timeout=Timeout.TIMEOUT_10MIN,
                    sleep=5,
                    func=get_job_status,
                    **common,
                    job_id=job_id,
                ):
                    state = sample.get("status", {}).get("state", "")
                    if state == "running":
                        break
                    if state in EVALHUB_JOB_TERMINAL_STATES:
                        LOGGER.info(f"Job {job_id} reached terminal state {state} before 'running' was observed")
                        break
            except TimeoutExpiredError:
                pytest.fail(f"EvalHub job {job_id} did not leave 'pending' state")

            job_result = wait_for_evalhub_job(**common, job_id=job_id, timeout=Timeout.TIMEOUT_10MIN)
            assert job_result["status"]["state"] == "completed", (
                f"Expected job {job_id} to complete, got: {job_result['status']}"
            )
        finally:
            if job_id:
                cleanup_evalhub_job(**common, job_id=job_id)
