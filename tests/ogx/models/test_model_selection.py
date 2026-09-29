import pytest

from tests.ogx.utils import select_ogx_model


class DummyModel:
    def __init__(self, model_id: str, model_type: str = "llm"):
        self.id = model_id
        self.custom_metadata = {"model_type": model_type}


class DummyProvider:
    def __init__(self, provider_id: str):
        self.provider_id = provider_id


@pytest.mark.parametrize(
    ("configured_model", "available_models", "expected_model_id"),
    [
        # Exact match preferred over substring or position
        (
            "Qwen3.8-27B",
            [
                DummyModel(model_id="vllm-inference/Qwen3.8-27B-Vision"),
                DummyModel(model_id="Qwen3.8-27B"),
            ],
            "Qwen3.8-27B",
        ),
        # Substring fallback when exact match is not found
        (
            "Qwen3.8-27B",
            [
                DummyModel(model_id="vllm-inference/Other-Vision-3.2"),
                DummyModel(model_id="vllm-inference/Qwen3.8-27B"),
            ],
            "vllm-inference/Qwen3.8-27B",
        ),
        # Unmatched configured model falls back to qwen model or first non-vision LLM
        (
            "NonExistentModel",
            [
                DummyModel(model_id="vllm-inference/Other-Vision-3.2"),
                DummyModel(model_id="vllm-inference/Qwen3.8-27B"),
            ],
            "vllm-inference/Qwen3.8-27B",
        ),
        # No configured model selects non-vision qwen model over qwen vision model
        (
            "",
            [
                DummyModel(model_id="vllm-inference/Qwen2.5-VL-7B-Instruct"),
                DummyModel(model_id="vllm-inference/Qwen3.8-27B"),
            ],
            "vllm-inference/Qwen3.8-27B",
        ),
        # No configured model selects qwen model if present
        (
            "",
            [
                DummyModel(model_id="vllm-inference/Other-Vision-3.2"),
                DummyModel(model_id="vllm-inference/Qwen3.8-27B"),
            ],
            "vllm-inference/Qwen3.8-27B",
        ),
        # Vision-only models fall back to first LLM model
        (
            "",
            [
                DummyModel(model_id="vllm-inference/Other-Vision-3.2"),
            ],
            "vllm-inference/Other-Vision-3.2",
        ),
    ],
)
def test_ogx_models_selection(
    configured_model: str,
    available_models: list[DummyModel],
    expected_model_id: str,
) -> None:
    mock_providers = [DummyProvider(provider_id="sentence-transformers")]

    embedding_model = DummyModel(model_id="sentence-transformers/all-MiniLM-L6-v2", model_type="embedding")
    embedding_model.custom_metadata["embedding_dimension"] = 384
    embedding_model.custom_metadata["provider_id"] = "sentence-transformers"
    available_models.append(embedding_model)

    result = select_ogx_model(
        models=available_models,
        providers=mock_providers,
        configured_model=configured_model,
    )
    assert result.model_id == expected_model_id


def test_ogx_models_selection_no_llm_raises_value_error() -> None:
    mock_providers = [DummyProvider(provider_id="sentence-transformers")]
    embedding_model = DummyModel(model_id="sentence-transformers/all-MiniLM-L6-v2", model_type="embedding")
    embedding_model.custom_metadata["embedding_dimension"] = 384
    embedding_model.custom_metadata["provider_id"] = "sentence-transformers"

    with pytest.raises(ValueError, match="No LLM models found in OGX client"):
        select_ogx_model(
            models=[embedding_model],
            providers=mock_providers,
            configured_model="",
        )
