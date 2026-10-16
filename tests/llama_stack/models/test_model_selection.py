import pytest

from tests.llama_stack.utils import select_llama_stack_model


class DummyModel:
    def __init__(self, model_id: str, model_type: str = "llm") -> None:
        self.id = model_id
        self.custom_metadata = {"model_type": model_type}


class DummyProvider:
    def __init__(self, provider_id: str) -> None:
        self.provider_id = provider_id


def _embedding_model() -> DummyModel:
    embedding_model = DummyModel(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        model_type="embedding",
    )
    embedding_model.custom_metadata["embedding_dimension"] = 384
    embedding_model.custom_metadata["provider_id"] = "sentence-transformers"
    return embedding_model


@pytest.mark.llama_stack
@pytest.mark.parametrize(
    ("configured_model", "available_models", "expected_model_id"),
    [
        pytest.param(
            "Qwen3.8-27B",
            [
                DummyModel(model_id="vllm-inference/Qwen3.8-27B-Vision"),
                DummyModel(model_id="Qwen3.8-27B"),
            ],
            "Qwen3.8-27B",
            id="test_exact_match_preferred_over_vision_model",
        ),
        pytest.param(
            "Qwen3.8-27B",
            [
                DummyModel(model_id="vllm-inference/Other-Vision-3.2"),
                DummyModel(model_id="vllm-inference/Qwen3.8-27B"),
            ],
            "vllm-inference/Qwen3.8-27B",
            id="test_substring_match_used_when_exact_match_missing",
        ),
        pytest.param(
            "NonExistentModel",
            [
                DummyModel(model_id="vllm-inference/Other-Vision-3.2"),
                DummyModel(model_id="vllm-inference/Qwen3.8-27B"),
            ],
            "vllm-inference/Qwen3.8-27B",
            id="test_unmatched_model_falls_back_to_qwen",
        ),
        pytest.param(
            "",
            [
                DummyModel(model_id="vllm-inference/Qwen2.5-VL-7B-Instruct"),
                DummyModel(model_id="vllm-inference/Qwen3.8-27B"),
            ],
            "vllm-inference/Qwen3.8-27B",
            id="test_no_configured_model_excludes_qwen_vision_model",
        ),
        pytest.param(
            "",
            [
                DummyModel(model_id="vllm-inference/Other-Vision-3.2"),
                DummyModel(model_id="vllm-inference/Qwen3.8-27B"),
            ],
            "vllm-inference/Qwen3.8-27B",
            id="test_no_configured_model_selects_qwen_model",
        ),
        pytest.param(
            "",
            [DummyModel(model_id="vllm-inference/Other-Vision-3.2")],
            "vllm-inference/Other-Vision-3.2",
            id="test_vision_only_models_fall_back_to_first_llm",
        ),
    ],
)
def test_llama_stack_model_selection(
    configured_model: str,
    available_models: list[DummyModel],
    expected_model_id: str,
) -> None:
    """Given available models, When selecting a model, Then the configured or safest LLM is returned."""
    providers = [DummyProvider(provider_id="sentence-transformers")]
    models = [*available_models, _embedding_model()]

    result = select_llama_stack_model(
        models=models,
        providers=providers,
        configured_model=configured_model,
    )

    assert result.model_id == expected_model_id


@pytest.mark.llama_stack
def test_llama_stack_model_selection_no_llm_raises_value_error() -> None:
    """Given only an embedding model, When selecting a model, Then a ValueError is raised."""
    providers = [DummyProvider(provider_id="sentence-transformers")]

    with pytest.raises(ValueError, match="No LLM models found in LlamaStack client"):
        select_llama_stack_model(
            models=[_embedding_model()],
            providers=providers,
            configured_model="",
        )
