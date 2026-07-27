import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.job import Job
from ocp_resources.namespace import Namespace
from ocp_resources.resource import ResourceEditor
from ocp_resources.service import Service
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_safety.evalhub.utils import (
    build_evalhub_kueue_job_payload,
    cleanup_evalhub_job,
    evalhub_runtime_label_selector,
    get_evalhub_job_workload,
    log_job_kueue_labels,
    submit_evalhub_job,
    wait_for_evalhub_job_workload_admitted,
    wait_for_evalhub_job_workload_inadmissible,
)
from utilities.constants import Timeout
from utilities.kueue_utils import ClusterQueue, LocalQueue, check_gated_pods_and_running_pods

LOGGER = structlog.get_logger(name=__name__)


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
            func=get_evalhub_job_workload,
            admin_client=admin_client,
            namespace=namespace,
            evalhub_job_id=evalhub_job_id,
        ):
            if workload is None:
                return
    except TimeoutExpiredError:
        raise TimeoutExpiredError(f"Kueue Workload for job {evalhub_job_id} still present after {timeout}s") from None


@pytest.mark.kueue
@pytest.mark.tier2
class TestEvalHubKueueJobDeletion:
    """Verify Kueue Workloads are cleaned up when EvalHub jobs are deleted."""

    def test_delete_pending_job_cleans_workload(
        self,
        admin_client: DynamicClient,
        evalhub_kueue_namespace: Namespace,
        evalhub_kueue_single_job_local_queue: LocalQueue,
        evalhub_kueue_vllm_service: Service,
        evalhub_kueue_request_common: dict[str, str],
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
        common = evalhub_kueue_request_common

        cluster_queue_name = evalhub_kueue_single_job_local_queue.instance.spec.clusterQueue
        cluster_queue = ClusterQueue(client=admin_client, name=cluster_queue_name)
        job_id = None

        try:
            with ResourceEditor(patches={cluster_queue: {"spec": {"stopPolicy": "HoldAndDrain"}}}):
                data = submit_evalhub_job(
                    **common,
                    payload=build_evalhub_kueue_job_payload(
                        queue_name=evalhub_kueue_single_job_local_queue.name,
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
                        timeout=Timeout.TIMEOUT_10MIN,
                    )
                except TimeoutExpiredError:
                    log_job_kueue_labels(admin_client, evalhub_kueue_namespace.name, job_id)
                    raise

                _delete_k8s_job(
                    admin_client=admin_client, namespace=evalhub_kueue_namespace.name, evalhub_job_id=job_id
                )

            _wait_for_workload_absent(
                admin_client=admin_client,
                namespace=evalhub_kueue_namespace.name,
                evalhub_job_id=job_id,
            )
        finally:
            if job_id:
                cleanup_evalhub_job(**common, job_id=job_id)

    def test_delete_running_job_cleans_workload(
        self,
        admin_client: DynamicClient,
        evalhub_kueue_namespace: Namespace,
        evalhub_kueue_multi_job_local_queue: LocalQueue,
        evalhub_kueue_vllm_service: Service,
        evalhub_kueue_request_common: dict[str, str],
    ) -> None:
        """TC-DEL-002: Deleting an admitted (running) job's K8s Job removes its Workload.

        Kueue must release the reserved quota when a job's K8s Job object is
        deleted mid-execution, allowing other workloads to be admitted.

        Note: Uses admin-level K8s Job deletion because the EvalHub API's
        per-job DELETE path is not covered by the operator-generated auth.yaml.
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
                    job_name="tc-del-002",
                ),
            )
            job_id = data["resource"]["id"]

            try:
                wait_for_evalhub_job_workload_admitted(
                    admin_client=admin_client,
                    namespace=evalhub_kueue_namespace.name,
                    evalhub_job_id=job_id,
                    timeout=Timeout.TIMEOUT_10MIN,
                )
            except TimeoutExpiredError:
                log_job_kueue_labels(admin_client, evalhub_kueue_namespace.name, job_id)
                raise

            selector = evalhub_runtime_label_selector(evalhub_job_id=job_id)
            try:
                for running, _ in TimeoutSampler(
                    wait_timeout=Timeout.TIMEOUT_10MIN,
                    sleep=5,
                    func=check_gated_pods_and_running_pods,
                    labels=[selector],
                    namespace=evalhub_kueue_namespace.name,
                    admin_client=admin_client,
                ):
                    if running >= 1:
                        break
            except TimeoutExpiredError:
                pytest.fail(f"Pod for job {job_id} did not reach running state within {Timeout.TIMEOUT_10MIN}s")

            _delete_k8s_job(admin_client=admin_client, namespace=evalhub_kueue_namespace.name, evalhub_job_id=job_id)

            _wait_for_workload_absent(
                admin_client=admin_client,
                namespace=evalhub_kueue_namespace.name,
                evalhub_job_id=job_id,
                timeout=120,
            )
        finally:
            if job_id:
                cleanup_evalhub_job(**common, job_id=job_id)
