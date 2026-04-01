import logging
from typing import Any, Dict

from kubernetes.dynamic import DynamicClient
from ocp_resources.pod import Pod
from timeout_sampler import TimeoutSampler

logger = logging.getLogger(__name__)


def wait_for_eval_job_completion(
    llama_stack_client: Any,
    job_id: str,
    benchmark_id: str,
    wait_timeout: int = 600,
    sleep: int = 30,
) -> None:
    """
    Wait for a LlamaStack eval job to complete.

    Args:
        llama_stack_client: The LlamaStack client instance
        job_id: The ID of the eval job to monitor
        benchmark_id: The ID of the benchmark being evaluated
        wait_timeout: Maximum time to wait in seconds (default: 600)
        sleep: Time to sleep between status checks in seconds (default: 30)
    """

    def _get_job_status() -> str:
        job = llama_stack_client.alpha.eval.jobs.status(job_id=job_id, benchmark_id=benchmark_id)
        status = job.status
        metadata = getattr(job, "metadata", None)
        logger.info("Job %s status=%s metadata=%s", job_id, status, metadata)
        return status

    samples = TimeoutSampler(
        wait_timeout=wait_timeout,
        sleep=sleep,
        func=_get_job_status,
    )

    for sample in samples:
        if sample == "completed":
            break
        elif sample in ("failed", "cancelled"):
            final_job = llama_stack_client.alpha.eval.jobs.status(
                job_id=job_id, benchmark_id=benchmark_id
            )
            metadata = getattr(final_job, "metadata", {})
            raise RuntimeError(
                f"Eval job {job_id} for benchmark {benchmark_id} terminated with "
                f"status: {sample}. Metadata: {metadata}"
            )


def validate_eval_result_structure(result: Any) -> Dict[str, Any]:
    """
    Validate that an EvaluateResponse has the expected Garak result structure.

    Asserts that ``generations`` is a non-empty list, ``scores`` is a dict containing
    an ``_overall`` key, and that ``_overall.aggregated_results`` contains the expected
    metric fields.

    Args:
        result: The EvaluateResponse object returned by job_result()

    Returns:
        The ``_overall`` aggregated_results dict for further assertions

    Raises:
        AssertionError: If the result structure is invalid
    """
    assert hasattr(result, "generations"), "Result missing 'generations' attribute"
    assert isinstance(result.generations, list), "generations must be a list"
    assert len(result.generations) > 0, "generations must not be empty"

    assert hasattr(result, "scores"), "Result missing 'scores' attribute"
    assert isinstance(result.scores, dict), "scores must be a dict"
    assert "_overall" in result.scores, "scores must contain '_overall' key"

    overall = result.scores["_overall"]
    assert hasattr(overall, "aggregated_results"), "_overall missing 'aggregated_results'"

    aggregated: Dict[str, Any] = overall.aggregated_results
    assert isinstance(aggregated, dict), "aggregated_results must be a dict"
    assert "total_attempts" in aggregated, "aggregated_results missing 'total_attempts'"
    assert "vulnerable_responses" in aggregated, "aggregated_results missing 'vulnerable_responses'"
    assert "attack_success_rate" in aggregated, "aggregated_results missing 'attack_success_rate'"
    assert aggregated["total_attempts"] > 0, "total_attempts must be > 0"
    assert isinstance(aggregated["attack_success_rate"], (int, float)), "attack_success_rate must be numeric"

    return aggregated


def validate_job_metadata(job: Any) -> Dict[str, Any]:
    """
    Validate that a Garak job response contains expected KFP metadata.

    Args:
        job: The Job object returned by run_eval() or job_status()

    Returns:
        The job metadata dict for further assertions

    Raises:
        AssertionError: If the job metadata is missing expected fields
    """
    assert hasattr(job, "metadata"), "Job missing 'metadata' attribute"
    metadata: Dict[str, Any] = job.metadata
    assert isinstance(metadata, dict), "metadata must be a dict"
    assert "kfp_run_id" in metadata, "metadata missing 'kfp_run_id'"
    assert "created_at" in metadata, "metadata missing 'created_at'"
    assert metadata["kfp_run_id"], "kfp_run_id must not be empty"
    assert metadata["created_at"], "created_at must not be empty"

    return metadata


def wait_for_dspa_pods(admin_client: DynamicClient, namespace: str, dspa_name: str, timeout: int = 300) -> None:
    """
    Wait for all DataSciencePipelinesApplication pods to be running.

    Args:
        admin_client: The admin client to use for pod retrieval
        namespace: The namespace where DSPA is deployed
        dspa_name: The name of the DSPA resource
        timeout: Timeout in seconds
    """

    label_selector = f"dspa={dspa_name}"

    def _all_dspa_pods_running() -> bool:
        pods = list(Pod.get(client=admin_client, namespace=namespace, label_selector=label_selector))
        if not pods:
            return False
        return all(pod.instance.status.phase == Pod.Status.RUNNING for pod in pods)

    sampler = TimeoutSampler(
        wait_timeout=timeout,
        sleep=10,
        func=_all_dspa_pods_running,
    )

    for is_ready in sampler:
        if is_ready:
            return
