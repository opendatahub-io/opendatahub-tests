import pytest
from simple_logger.logger import get_logger

from tests.model_explainability.constants import MNT_MODELS
from utilities.constants import MinIo

LOGGER = get_logger(name=__name__)
PII_REGEX_SHIELD_ID = "regex"


@pytest.mark.parametrize(
    "model_namespace, minio_pod, minio_data_connection",
    [
        pytest.param(
            {"name": "test-llamastack-lls"},
            MinIo.PodConfig.QWEN_MINIO_CONFIG,
            {"bucket": "llms"},
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

    def test_lmeval_register_benchmark(self, llamastack_client):
        llamastack_client.models.register(provider_id="vllm-inference", model_type="llm", model_id=MNT_MODELS)

        provider_id = "trustyai_lmeval"
        trustyai_lmeval_arc_easy = f"{provider_id}::arc_easy"
        llamastack_client.benchmarks.register(
            benchmark_id=trustyai_lmeval_arc_easy,
            dataset_id=trustyai_lmeval_arc_easy,
            scoring_functions=["string"],
            provider_id="trustyai_lmeval",
            metadata={"tokenized_request": False, "tokenizer": "google/flan-t5-small"},
        )

        benchmarks = llamastack_client.benchmarks.list()

        assert len(benchmarks) == 1
        assert benchmarks[0].identifier == trustyai_lmeval_arc_easy
        assert benchmarks[0].provider_id == provider_id
