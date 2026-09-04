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
    SINGLE_POD_QUOTA,
    VLLM_EMULATOR_IMAGE,
)
from tests.ai_safety.evalhub.utils import (
    cleanup_evalhub_job,
    cleanup_evalhub_jobs_and_workloads,
    evalhub_runtime_label_selector,
    log_job_kueue_labels,
    submit_evalhub_kueue_job_and_get_id,
    validate_evalhub_job_completed,
    wait_for_evalhub_job,
    wait_for_evalhub_job_workload_admitted,
    wait_for_evalhub_pod_started,
)
from utilities.constants import Timeout
from utilities.kueue_utils import (
    LocalQueue,
    Workload,
    build_resource_groups,
    create_cluster_queue,
    create_kueue_managed_job,
    create_local_queue,
    create_resource_flavor,
    create_workload_priority_class,
    wait_for_queue_active,
    wait_for_workload_condition,
    workload_has_condition,
)

LOGGER = structlog.get_logger(name=__name__)

# CPU/memory hold both jobs; the pods=1 cap is what forces preemption.
PREEMPTION_QUOTAS = {"cpu": KUEUE_CPU_QUOTA, "memory": KUEUE_MEMORY_QUOTA, "pods": SINGLE_POD_QUOTA}


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

        EvalHub users can't raise their own priority through the API, so eval jobs
        are always the lowest-priority (most preemptable) workloads in a queue.
        """
        common = evalhub_kueue_request_common
        job_id = None

        try:
            job_id = submit_evalhub_kueue_job_and_get_id(
                request_common=common,
                local_queue_name=evalhub_kueue_multi_job_local_queue.name,
                model_service_name=evalhub_kueue_vllm_service.name,
                tenant_namespace=evalhub_kueue_namespace.name,
                job_name="tc-prio-001-priority",
            )

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

        Checks that an EvalHub job is suspended when a higher-priority workload
        shows up, then resumes and runs to completion (from the start, since eval
        jobs don't checkpoint) once that workload is gone.
        """
        common = evalhub_kueue_request_common
        namespace = evalhub_kueue_namespace.name

        with (
            create_resource_flavor(name="tc-preempt-flavor", client=admin_client) as flavor,
            create_cluster_queue(
                name="tc-preempt-cq",
                client=admin_client,
                resource_groups=build_resource_groups(flavor_name=flavor.name, quotas=PREEMPTION_QUOTAS),
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
                victim_id = submit_evalhub_kueue_job_and_get_id(
                    request_common=common,
                    local_queue_name=local_queue.name,
                    model_service_name=evalhub_kueue_vllm_service.name,
                    tenant_namespace=namespace,
                    job_name="tc-preempt-001-victim",
                )
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

                wait_for_evalhub_pod_started(admin_client=admin_client, namespace=namespace, evalhub_job_id=victim_id)

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
                            workload_has_condition(workload=wl, condition_type="Evicted")
                            or workload_has_condition(workload=wl, condition_type="Preempted")
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
                cleanup_evalhub_jobs_and_workloads(
                    request_common=common, admin_client=admin_client, namespace=namespace, job_ids=job_ids
                )

    def test_evalhub_job_not_preempted_when_preemption_disabled(
        self,
        admin_client: DynamicClient,
        evalhub_kueue_namespace: Namespace,
        evalhub_kueue_vllm_service: Service,
        evalhub_kueue_request_common: dict[str, str],
    ) -> None:
        """TC-NOPREEMPT-001: With ``withinClusterQueue: Never``, eval jobs are not preempted.

        A dedicated queue with preemption disabled lets an eval job finish
        uninterrupted; a higher-priority competitor waits instead of evicting it.
        """
        common = evalhub_kueue_request_common
        namespace = evalhub_kueue_namespace.name

        with (
            create_resource_flavor(name="tc-nopreempt-flavor", client=admin_client) as flavor,
            create_cluster_queue(
                name="tc-nopreempt-cq",
                client=admin_client,
                resource_groups=build_resource_groups(flavor_name=flavor.name, quotas=PREEMPTION_QUOTAS),
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
                protected_id = submit_evalhub_kueue_job_and_get_id(
                    request_common=common,
                    local_queue_name=local_queue.name,
                    model_service_name=evalhub_kueue_vllm_service.name,
                    tenant_namespace=namespace,
                    job_name="tc-nopreempt-001-protected",
                )
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

                wait_for_evalhub_pod_started(
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
                    preemptor_uid = preemptor_job.instance.metadata.uid

                    def _preemptor_workload() -> Workload | None:
                        workloads = list(
                            Workload.get(
                                client=admin_client,
                                namespace=namespace,
                                label_selector=f"kueue.x-k8s.io/job-uid={preemptor_uid}",
                            )
                        )
                        return workloads[0] if workloads else None

                    # With preemption disabled the higher-priority competitor must be
                    # *actively refused* admission (QuotaReserved=False) rather than evict
                    # the running eval job. Waiting for an explicit False -- not merely a
                    # workload that lacks QuotaReserved=True yet -- avoids passing on an
                    # unreconciled workload that could still be admitted a moment later.
                    try:
                        for workload in TimeoutSampler(
                            wait_timeout=Timeout.TIMEOUT_5MIN,
                            sleep=5,
                            func=_preemptor_workload,
                        ):
                            if workload and workload_has_condition(
                                workload=workload, condition_type="QuotaReserved", status="False"
                            ):
                                break
                    except TimeoutExpiredError:
                        pytest.fail(
                            "High-priority preemptor never reported QuotaReserved=False; "
                            "expected it to be gated by withinClusterQueue=Never"
                        )

                    # The preemptor is gated precisely because the protected eval job is
                    # still holding the quota, so at this moment it must not have been
                    # evicted. (Once the protected job later finishes on its own, Kueue
                    # admitting the waiting preemptor is expected under ``Never`` and is not
                    # a violation -- hence we assert non-eviction here rather than requiring
                    # the preemptor to stay gated for the protected job's whole runtime.)
                    fresh_protected = Workload(client=admin_client, name=protected_workload.name, namespace=namespace)
                    assert not workload_has_condition(workload=fresh_protected, condition_type="Evicted"), (
                        "Protected EvalHub workload was evicted even though preemption is disabled"
                    )

                # Protected job completes uninterrupted.
                protected_result = wait_for_evalhub_job(**common, job_id=protected_id, timeout=Timeout.TIMEOUT_10MIN)
                validate_evalhub_job_completed(job_data=protected_result)
            finally:
                cleanup_evalhub_jobs_and_workloads(
                    request_common=common, admin_client=admin_client, namespace=namespace, job_ids=job_ids
                )
