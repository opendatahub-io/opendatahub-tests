import pytest
import yaml

from tests.model_explainability.guardrails.constants import BUILTIN_DETECTOR_CONFIG, CHAT_GENERATION_CONFIG
from tests.model_explainability.guardrails.test_guardrails import MNT_MODELS
from utilities.constants import MinIo


@pytest.mark.parametrize(
    "model_namespace, minio_pod, minio_data_connection, orchestrator_config, guardrails_orchestrator",
    [
        pytest.param(
            {"name": "test-llamastack-lmeval"},
            MinIo.PodConfig.QWEN_MINIO_CONFIG,
            {"bucket": "llms"},
            {
                "orchestrator_config_data": {
                    "config.yaml": yaml.dump({
                        "chat_generation": CHAT_GENERATION_CONFIG,
                        "detectors": BUILTIN_DETECTOR_CONFIG,
                    })
                },
            },
            {"enable_built_in_detectors": True, "enable_guardrails_gateway": False},
        )
    ],
    indirect=True,
)
class TestLlamaStackLMEvalProvider:
    def test_lmeval_register_model(self, qwen_isvc, llamastack_client_trustyai):
        provider_id = "vllm-inference"
        model_type = "llm"
        llamastack_client_trustyai.models.register(provider_id=provider_id, model_type=model_type, model_id=MNT_MODELS)
        models = llamastack_client_trustyai.models.list()

        # We only need to check the first model;
        # second is a granite embedding model present by default
        assert len(models) == 2
        assert models[0].identifier == "mnt/models"
        assert models[0].provider_id == "vllm-inference"
        assert models[0].model_type == "llm"

    def test_lmeval_register_benchmark(self, llamastack_client_trustyai):
        trustyai_lmeval_arc_easy = "trustyai_lmeval::arc_easy"
        llamastack_client_trustyai.benchmarks.register(
            benchmark_id=trustyai_lmeval_arc_easy,
            dataset_id=trustyai_lmeval_arc_easy,
            scoring_functions=["string"],
            provider_id="trustyai_lmeval",
            metadata={
                "tokenized_request": False,
                "tokenizer": "google/flan-t5-small"
            }
        )

        response = llamastack_client_trustyai.benchmarks.list()
        print(response)