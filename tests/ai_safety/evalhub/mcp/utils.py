from tests.ai_safety.evalhub.constants import EVALHUB_VLLM_EMULATOR_PORT

DEFAULT_TOKENIZER: str = "google/flan-t5-small"


def build_mcp_evaluation_arguments(
    model_service_name: str,
    tenant_namespace: str,
    *,
    job_name: str = "evalhub-mcp-e2e-job",
    benchmark_id: str = "arc_easy",
    provider_id: str = "lm_evaluation_harness",
    num_examples: int = 10,
    tokenizer: str = DEFAULT_TOKENIZER,
) -> dict:
    """Build MCP evaluation submission arguments for EvalHub.

    Constructs the payload expected by the EvalHub MCP ``submit_evaluation``
    tool.  The ``benchmark.parameters`` block **must** include a ``tokenizer``
    key so that the job execution framework treats the model name as an
    OpenAI-compatible endpoint identifier rather than attempting to load it
    as a HuggingFace model.

    Args:
        model_service_name: Kubernetes Service name for the vLLM emulator.
        tenant_namespace: Namespace where the model service runs.
        job_name: Human-readable name for the evaluation job.
        benchmark_id: Benchmark identifier (default ``arc_easy``).
        provider_id: Evaluation provider (default ``lm_evaluation_harness``).
        num_examples: Number of evaluation examples to run.
        tokenizer: HuggingFace tokenizer identifier required by
            OpenAI-compatible model endpoints.

    Returns:
        Dict suitable for posting to the EvalHub MCP ``submit_evaluation``
        tool or the ``/evaluations/jobs`` REST endpoint.
    """
    model_url = f"http://{model_service_name}.{tenant_namespace}.svc.cluster.local:{EVALHUB_VLLM_EMULATOR_PORT}/v1"
    return {
        "name": job_name,
        "model": {
            "url": model_url,
            "name": "emulatedModel",
        },
        "benchmarks": [
            {
                "id": benchmark_id,
                "provider_id": provider_id,
                "parameters": {
                    "num_examples": num_examples,
                    "tokenizer": tokenizer,
                },
            },
        ],
    }
