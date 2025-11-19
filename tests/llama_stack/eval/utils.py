from kubernetes.dynamic import DynamicClient
from ocp_resources.pod import Pod
from timeout_sampler import TimeoutSampler


def wait_for_eval_job_completion(
    llama_stack_client, job_id: str, benchmark_id: str, wait_timeout: int = 600, sleep: int = 30
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
    samples = TimeoutSampler(
        wait_timeout=wait_timeout,
        sleep=sleep,
        func=lambda: llama_stack_client.alpha.eval.jobs.status(job_id=job_id, benchmark_id=benchmark_id).status,
    )

    for sample in samples:
        if sample == "completed":
            break
        elif sample in ("failed", "cancelled"):
            raise RuntimeError(f"Eval job {job_id} for benchmark {benchmark_id} terminated with status: {sample}")


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
        pods = list(Pod.get(dyn_client=admin_client, namespace=namespace, label_selector=label_selector))
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
