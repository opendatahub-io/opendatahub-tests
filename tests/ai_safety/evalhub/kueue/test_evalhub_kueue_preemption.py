import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.job import Job
from ocp_resources.namespace import Namespace
from ocp_resources.service import Service
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_safety.evalhub.kueue.constants import (
    HIGH_PRIORITY_VALUE,
    KUEUE_CPU_QUOTA,
    KUEUE_MEMORY_QUOTA,
    PREEMPTOR_CPU_REQUEST,
    PREEMPTOR_MEMORY_REQUEST,
    VLLM_EMULATOR_IMAGE,
)
from tests.ai_safety.evalhub.utils import (
    build_evalhub_kueue_job_payload,
    cleanup_evalhub_job,
    evalhub_runtime_label_selector,
    get_evalhub_job_workload,
    log_job_kueue_labels,
    submit_evalhub_job,
    validate_evalhub_job_completed,
    wait_for_evalhub_job,
    wait_for_evalhub_job_workload_absent,
    wait_for_evalhub_job_workload_admitted,
)
from utilities.constants import Timeout
from utilities.kueue_utils import (
    LocalQueue,
    Workload,
    count_pods_started,
    create_cluster_queue,
    create_kueue_managed_job,
    create_local_queue,
    create_resource_flavor,
    create_workload_priority_class,
    wait_for_queue_active,
    wait_for_workload_condition,
)

LOGGER = structlog.get_logger(name=__name__)


def _single_job_resource_groups(flavor_name: str) -> list[dict]:
    """ResourceGroups sized so exactly one EvalHub job fits the quota."""
    return [
        {
            "coveredResources": ["cpu", "memory"],
            "flavors": [
                {
                    "name": flavor_name,
                    "resources": [
                        {"name": "cpu", "nominalQuota": KUEUE_CPU_QUOTA},
                        {"name": "memory", "nominalQuota": KUEUE_MEMORY_QUOTA},
                    ],
                }
            ],
        }
    ]


def _workload_condition(workload: Workload, condition_type: str) -> dict | None:
    """Return the named condition dict from a Workload, or None."""
    conditions = (workload.instance.status or {}).get("conditions", [])
    return next((c for c in conditions if c.get("type") == condition_type), None)


def _workload_has_condition(workload: Workload, condition_type: str, status: str = "True") -> bool:
    """True if the Workload carries condition_type with the given status."""
    condition = _workload_condition(workload=workload, condition_type=condition_type)
    return bool(condition and condition.get("status") == status)


def _wait_for_evalhub_pod_started(
    admin_client: DynamicClient,
    namespace: str,
    evalhub_job_id: str,
    timeout: int = Timeout.TIMEOUT_10MIN,
) -> None:
    """Wait until the EvalHub job's pod has actually started running.

    Preemption is only meaningful once the victim is occupying quota, so the
    preemptor must be created after the victim's pod has started.
    """
    selector = evalhub_runtime_label_selector(evalhub_job_id=evalhub_job_id)
    try:
        for started in TimeoutSampler(
            wait_timeout=timeout,
            sleep=5,
            func=count_pods_started,
            labels=[selector],
            namespace=namespace,
            admin_client=admin_client,
        ):
            if started >= 1:
                return
    except TimeoutExpiredError:
        pytest.fail(f"EvalHub job {evalhub_job_id} pod never started; cannot exercise preemption")


def _cleanup_evalhub_jobs(
    common: dict[str, str],
    admin_client: DynamicClient,
    namespace: str,
    job_ids: list[str],
) -> None:
    """Hard-delete EvalHub jobs and wait for their Kueue Workloads to disappear.

    Must run *before* any isolated ClusterQueue is torn down: a ClusterQueue
    holds a ``kueue.x-k8s.io/resource-in-use`` finalizer while Workloads still
    reference it, so the queue's deletion blocks (for minutes) until the jobs'
    Workloads are gone.
    """
    for jid in job_ids:
        workload = get_evalhub_job_workload(admin_client=admin_client, namespace=namespace, evalhub_job_id=jid)
        workload_name = workload.name if workload else None
        try:
            cleanup_evalhub_job(**common, job_id=jid)
        except Exception:
            LOGGER.warning(f"Failed to clean up job {jid}", exc_info=True)
            continue
        if workload_name:
            try:
                wait_for_evalhub_job_workload_absent(
                    admin_client=admin_client,
                    namespace=namespace,
                    workload_name=workload_name,
                    timeout=Timeout.TIMEOUT_2MIN,
                )
            except TimeoutExpiredError:
                LOGGER.warning(f"Workload {workload_name} still present after cleaning up job {jid}", exc_info=True)


@pytest.mark.kueue
@pytest.mark.tier2
class TestEvalHubKueuePriority:
    """Verify the priority the EvalHub API assigns to Kueue workloads."""

    def test_evalhub_job_workload_has_default_priority_zero(
        self,
        admin_client: DynamicClient,
        evalhub_kueue_namespace: Namespace,
        evalhub_kueue_multi_job_local_queue: LocalQueue,
        evalhub_kueue_vllm_service: Service,
        evalhub_kueue_request_common: dict[str, str],
    ) -> None:
        """TC-PRIO-001: EvalHub-submitted jobs carry priority 0 and no priority class.

        The blog documents that "Jobs submitted via the EvalHub API are assigned
        priority 0 by default." All preemption behavior rests on this: EvalHub
        users cannot raise their own priority through the API, so eval jobs are
        always the lowest-priority (most preemptable) workloads in a queue.
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
                    job_name="tc-prio-001-priority",
                ),
            )
            job_id = data["resource"]["id"]

            workload = wait_for_evalhub_job_workload_admitted(
                admin_client=admin_client,
                namespace=evalhub_kueue_namespace.name,
                evalhub_job_id=job_id,
                timeout=Timeout.TIMEOUT_10MIN,
            )

            spec = workload.instance.spec or {}
            priority = spec.get("priority")
            assert priority == 0, f"Expected EvalHub workload priority 0, got {priority}"

            priority_class_name = spec.get("priorityClassName")
            assert not priority_class_name, (
                f"Expected no priorityClassName on an EvalHub workload, got {priority_class_name!r}"
            )
        finally:
            if job_id:
                cleanup_evalhub_job(**common, job_id=job_id)


@pytest.mark.kueue
@pytest.mark.tier2
class TestEvalHubKueuePreemption:
    """Verify preemption behavior of EvalHub jobs under Kueue admission control."""

    def test_evalhub_job_preempted_by_higher_priority_workload(
        self,
        admin_client: DynamicClient,
        evalhub_kueue_namespace: Namespace,
        evalhub_kueue_vllm_service: Service,
        evalhub_kueue_request_common: dict[str, str],
    ) -> None:
        """TC-PREEMPT-001: A running EvalHub job is preempted, then resumes and completes.

        With ``preemption.withinClusterQueue: LowerPriority`` a higher-priority
        workload that needs the whole quota evicts the running (priority 0)
        EvalHub job. The blog documents the observable evidence: the Workload
        gains ``Evicted``/``Preempted`` conditions and the batch Job flips to
        ``Suspended=True``. Once the preemptor is removed, Kueue re-admits the
        gated EvalHub job and it runs to completion (from the start — eval jobs
        do not checkpoint).

        Uses an isolated ClusterQueue so the session-scoped shared queues are
        never mutated. Timing note: the victim must still be running when the
        preemptor is admitted; the pod-started gate below makes that reliable.
        """
        common = evalhub_kueue_request_common
        namespace = evalhub_kueue_namespace.name

        with (
            create_resource_flavor(name="tc-preempt-flavor", client=admin_client) as flavor,
            create_cluster_queue(
                name="tc-preempt-cq",
                client=admin_client,
                resource_groups=_single_job_resource_groups(flavor_name=flavor.name),
                namespace_selector={},
                preemption={"withinClusterQueue": "LowerPriority"},
            ) as cluster_queue,
            create_local_queue(
                name="tc-preempt-lq",
                namespace=namespace,
                cluster_queue=cluster_queue.name,
                client=admin_client,
            ) as local_queue,
            create_workload_priority_class(
                name="tc-preempt-high-priority",
                value=HIGH_PRIORITY_VALUE,
                client=admin_client,
            ) as high_priority_class,
        ):
            wait_for_queue_active(queue=cluster_queue)
            wait_for_queue_active(queue=local_queue)

            # Clean up EvalHub jobs (and their Workloads) before the queues tear
            # down; a ClusterQueue's resource-in-use finalizer blocks deletion
            # while any Workload still references it.
            job_ids: list[str] = []
            try:
                data = submit_evalhub_job(
                    **common,
                    payload=build_evalhub_kueue_job_payload(
                        queue_name=local_queue.name,
                        model_service_name=evalhub_kueue_vllm_service.name,
                        tenant_namespace=namespace,
                        job_name="tc-preempt-001-victim",
                    ),
                )
                victim_id = data["resource"]["id"]
                job_ids.append(victim_id)

                try:
                    victim_workload = wait_for_evalhub_job_workload_admitted(
                        admin_client=admin_client,
                        namespace=namespace,
                        evalhub_job_id=victim_id,
                        timeout=Timeout.TIMEOUT_10MIN,
                    )
                except TimeoutExpiredError:
                    log_job_kueue_labels(admin_client, namespace, victim_id)
                    raise

                _wait_for_evalhub_pod_started(admin_client=admin_client, namespace=namespace, evalhub_job_id=victim_id)

                with create_kueue_managed_job(
                    client=admin_client,
                    name="tc-preempt-001-preemptor",
                    namespace=namespace,
                    local_queue=local_queue.name,
                    image=VLLM_EMULATOR_IMAGE,
                    cpu_request=PREEMPTOR_CPU_REQUEST,
                    memory_request=PREEMPTOR_MEMORY_REQUEST,
                    priority_class=high_priority_class.name,
                ):
                    # The victim workload should be evicted/preempted to make room.
                    wait_for_workload_condition(
                        client=admin_client,
                        workload_name=victim_workload.name,
                        namespace=namespace,
                        condition_check=lambda wl: (
                            _workload_has_condition(workload=wl, condition_type="Evicted")
                            or _workload_has_condition(workload=wl, condition_type="Preempted")
                        ),
                        condition_name="Evicted/Preempted",
                        timeout=Timeout.TIMEOUT_5MIN,
                    )

                    # The victim's batch Job should report Suspended=True.
                    selector = evalhub_runtime_label_selector(evalhub_job_id=victim_id)
                    victim_jobs = list(Job.get(client=admin_client, namespace=namespace, label_selector=selector))
                    assert victim_jobs, f"No batch Job found for preempted EvalHub job {victim_id}"
                    victim_job = victim_jobs[0]
                    try:
                        for job_sample in TimeoutSampler(
                            wait_timeout=Timeout.TIMEOUT_2MIN,
                            sleep=5,
                            func=lambda: Job(client=admin_client, name=victim_job.name, namespace=namespace),
                        ):
                            conditions = (job_sample.instance.status or {}).get("conditions", [])
                            if any(c.get("type") == "Suspended" and c.get("status") == "True" for c in conditions):
                                break
                    except TimeoutExpiredError:
                        pytest.fail(f"Preempted Job {victim_job.name} never reported Suspended=True")

                # Preemptor removed: the gated EvalHub job resumes and completes.
                victim_result = wait_for_evalhub_job(**common, job_id=victim_id, timeout=Timeout.TIMEOUT_10MIN)
                validate_evalhub_job_completed(job_data=victim_result)
            finally:
                _cleanup_evalhub_jobs(common=common, admin_client=admin_client, namespace=namespace, job_ids=job_ids)

    def test_evalhub_job_not_preempted_when_preemption_disabled(
        self,
        admin_client: DynamicClient,
        evalhub_kueue_namespace: Namespace,
        evalhub_kueue_vllm_service: Service,
        evalhub_kueue_request_common: dict[str, str],
    ) -> None:
        """TC-NOPREEMPT-001: With ``withinClusterQueue: Never``, eval jobs are not preempted.

        This validates the blog's operational recommendation: give evaluation
        jobs a dedicated ClusterQueue with ``withinClusterQueue: Never`` so that,
        because eval workloads cannot checkpoint, they complete without
        interruption. A higher-priority competitor is forced to wait rather than
        evict the running EvalHub job.
        """
        common = evalhub_kueue_request_common
        namespace = evalhub_kueue_namespace.name

        with (
            create_resource_flavor(name="tc-nopreempt-flavor", client=admin_client) as flavor,
            create_cluster_queue(
                name="tc-nopreempt-cq",
                client=admin_client,
                resource_groups=_single_job_resource_groups(flavor_name=flavor.name),
                namespace_selector={},
                preemption={"withinClusterQueue": "Never"},
            ) as cluster_queue,
            create_local_queue(
                name="tc-nopreempt-lq",
                namespace=namespace,
                cluster_queue=cluster_queue.name,
                client=admin_client,
            ) as local_queue,
            create_workload_priority_class(
                name="tc-nopreempt-high-priority",
                value=HIGH_PRIORITY_VALUE,
                client=admin_client,
            ) as high_priority_class,
        ):
            wait_for_queue_active(queue=cluster_queue)
            wait_for_queue_active(queue=local_queue)

            # The effective preemption policy is what the blog tells operators to set.
            effective_preemption = (cluster_queue.instance.spec or {}).get("preemption", {})
            assert effective_preemption.get("withinClusterQueue") == "Never", (
                f"Expected withinClusterQueue=Never, got: {effective_preemption}"
            )

            # Clean up EvalHub jobs (and their Workloads) before the queues tear
            # down; a ClusterQueue's resource-in-use finalizer blocks deletion
            # while any Workload still references it.
            job_ids: list[str] = []
            try:
                data = submit_evalhub_job(
                    **common,
                    payload=build_evalhub_kueue_job_payload(
                        queue_name=local_queue.name,
                        model_service_name=evalhub_kueue_vllm_service.name,
                        tenant_namespace=namespace,
                        job_name="tc-nopreempt-001-protected",
                    ),
                )
                protected_id = data["resource"]["id"]
                job_ids.append(protected_id)

                try:
                    protected_workload = wait_for_evalhub_job_workload_admitted(
                        admin_client=admin_client,
                        namespace=namespace,
                        evalhub_job_id=protected_id,
                        timeout=Timeout.TIMEOUT_10MIN,
                    )
                except TimeoutExpiredError:
                    log_job_kueue_labels(admin_client, namespace, protected_id)
                    raise

                _wait_for_evalhub_pod_started(
                    admin_client=admin_client, namespace=namespace, evalhub_job_id=protected_id
                )

                with create_kueue_managed_job(
                    client=admin_client,
                    name="tc-nopreempt-001-preemptor",
                    namespace=namespace,
                    local_queue=local_queue.name,
                    image=VLLM_EMULATOR_IMAGE,
                    cpu_request=PREEMPTOR_CPU_REQUEST,
                    memory_request=PREEMPTOR_MEMORY_REQUEST,
                    priority_class=high_priority_class.name,
                ) as preemptor_job:
                    # The high-priority preemptor must queue (QuotaReserved=False), not evict.
                    preemptor_uid = preemptor_job.instance.metadata.uid

                    def _preemptor_gated() -> bool:
                        workloads = list(
                            Workload.get(
                                client=admin_client,
                                namespace=namespace,
                                label_selector=f"kueue.x-k8s.io/job-uid={preemptor_uid}",
                            )
                        )
                        return bool(workloads) and not _workload_has_condition(
                            workload=workloads[0], condition_type="QuotaReserved"
                        )

                    try:
                        for gated in TimeoutSampler(
                            wait_timeout=Timeout.TIMEOUT_5MIN,
                            sleep=5,
                            func=_preemptor_gated,
                        ):
                            if gated:
                                break
                    except TimeoutExpiredError:
                        pytest.fail("High-priority preemptor was admitted despite withinClusterQueue=Never")

                    # The protected EvalHub workload must not have been evicted.
                    fresh_protected = Workload(client=admin_client, name=protected_workload.name, namespace=namespace)
                    assert not _workload_has_condition(workload=fresh_protected, condition_type="Evicted"), (
                        "Protected EvalHub workload was evicted even though preemption is disabled"
                    )

                # Protected job completes uninterrupted.
                protected_result = wait_for_evalhub_job(**common, job_id=protected_id, timeout=Timeout.TIMEOUT_10MIN)
                validate_evalhub_job_completed(job_data=protected_result)
            finally:
                _cleanup_evalhub_jobs(common=common, admin_client=admin_client, namespace=namespace, job_ids=job_ids)
