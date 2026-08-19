import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.resource import ResourceEditor
from ocp_resources.service import Service
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_safety.evalhub.utils import (
    build_evalhub_kueue_job_payload,
    cleanup_evalhub_job,
    cluster_queue_name,
    delete_evalhub_runtime_k8s_job,
    evalhub_runtime_label_selector,
    log_job_kueue_labels,
    submit_evalhub_job,
    wait_for_evalhub_job_workload_absent,
    wait_for_evalhub_job_workload_admitted,
    wait_for_evalhub_job_workload_inadmissible,
)
from utilities.constants import Timeout
from utilities.kueue_utils import ClusterQueue, LocalQueue, check_gated_pods_and_running_pods

LOGGER = structlog.get_logger(name=__name__)


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
                        job_name="tc-del-001",
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

                delete_evalhub_runtime_k8s_job(
                    admin_client=admin_client, namespace=evalhub_kueue_namespace.name, evalhub_job_id=job_id
                )

            wait_for_evalhub_job_workload_absent(
                admin_client=admin_client,
                namespace=evalhub_kueue_namespace.name,
                workload_name=workload.name,
            )
        finally:
            if job_id:
                try:
                    cleanup_evalhub_job(**common, job_id=job_id)
                except Exception:
                    LOGGER.warning(f"Failed to clean up job {job_id}", exc_info=True)

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
                workload = wait_for_evalhub_job_workload_admitted(
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

            delete_evalhub_runtime_k8s_job(
                admin_client=admin_client, namespace=evalhub_kueue_namespace.name, evalhub_job_id=job_id
            )

            wait_for_evalhub_job_workload_absent(
                admin_client=admin_client,
                namespace=evalhub_kueue_namespace.name,
                workload_name=workload.name,
                timeout=120,
            )
        finally:
            if job_id:
                try:
                    cleanup_evalhub_job(**common, job_id=job_id)
                except Exception:
                    LOGGER.warning(f"Failed to clean up job {job_id}", exc_info=True)
