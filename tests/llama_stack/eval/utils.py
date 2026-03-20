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
