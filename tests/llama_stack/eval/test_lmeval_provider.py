import pytest

from tests.llama_stack.constants import LlamaStackProviders


@pytest.mark.parametrize(
    "model_namespace, llama_stack_server_config",
    [
        pytest.param(
            {"name": "test-guardrails-lls"},
            {
                "vllm_url_fixture": "qwen25_05B_instruct_url",
                "inference_model": "qwen",
            },
        )
    ],
    indirect=True,
)
@pytest.mark.rawdeployment
class TestLlamaStackLMEvalProvider:
    """
    Adds basic tests for the LlamaStack LMEval provider.

    1. Register the LLM that will be evaluated.
    2. Register the arc_easy benchmark (eval)
    3. TODO: Add test for run_eval
    """

    def test_lmeval_register_benchmark(self, llama_stack_client):
        llama_stack_client.models.register(
            provider_id=LlamaStackProviders.Inference.VLLM_INFERENCE, model_type="llm", model_id="qwen"
        )

        provider_id = LlamaStackProviders.Eval.TRUSTYAI_LMEVAL
        trustyai_lmeval_arc_easy = f"{provider_id}::arc_easy"
        llama_stack_client.benchmarks.register(
            benchmark_id=trustyai_lmeval_arc_easy,
            dataset_id=trustyai_lmeval_arc_easy,
            scoring_functions=["string"],
            provider_id=LlamaStackProviders.Eval.TRUSTYAI_LMEVAL,
            provider_benchmark_id="string",
            metadata={"tokenized_requests": False, "tokenizer": "google/flan-t5-small"},
        )

        benchmarks = llama_stack_client.benchmarks.list()

        assert len(benchmarks) == 1
        assert benchmarks[0].identifier == trustyai_lmeval_arc_easy
        assert benchmarks[0].provider_id == provider_id

    def test_llamastack_run_eval(self, patched_trustyai_operator_configmap_allow_online, llama_stack_client):
        job = llama_stack_client.eval.run_eval(
            benchmark_id=f"{LlamaStackProviders.Eval.TRUSTYAI_LMEVAL}::arc_easy",
            benchmark_config={
                "eval_candidate": {
                    "model": "qwen",
                    "type": "model",
                    "provider_id": LlamaStackProviders.Eval.TRUSTYAI_LMEVAL,
                    "sampling_params": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 256},
                },
                "num_examples": 100,
            },
        )

        print("hi")
