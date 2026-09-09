from unittest.mock import MagicMock

import pytest

import tests.ogx.conftest as ogx_conftest


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
                DummyModel("vllm-inference/Qwen3.8-27B-Vision"),
                DummyModel("Qwen3.8-27B"),
            ],
            "Qwen3.8-27B",
        ),
        # Substring fallback when exact match is not found
        (
            "Qwen3.8-27B",
            [
                DummyModel("vllm-inference/Other-Vision-3.2"),
                DummyModel("vllm-inference/Qwen3.8-27B"),
            ],
            "vllm-inference/Qwen3.8-27B",
        ),
        # Unmatched configured model falls back to qwen model or first non-vision LLM
        (
            "NonExistentModel",
            [
                DummyModel("vllm-inference/Other-Vision-3.2"),
                DummyModel("vllm-inference/Qwen3.8-27B"),
            ],
            "vllm-inference/Qwen3.8-27B",
        ),
        # No configured model selects qwen model if present
        (
            "",
            [
                DummyModel("vllm-inference/Other-Vision-3.2"),
                DummyModel("vllm-inference/Qwen3.8-27B"),
            ],
            "vllm-inference/Qwen3.8-27B",
        ),
        # Vision-only models fall back to first LLM model
        (
            "",
            [
                DummyModel("vllm-inference/Other-Vision-3.2"),
            ],
            "vllm-inference/Other-Vision-3.2",
        ),
    ],
)
def test_ogx_models_selection(monkeypatch: pytest.MonkeyPatch, configured_model: str, available_models: list[DummyModel], expected_model_id: str) -> None:
    monkeypatch.setattr(ogx_conftest, "OGX_CORE_INFERENCE_MODEL", configured_model)

    mock_client = MagicMock()
    mock_client.models.list.return_value.data = available_models
    mock_client.providers.list.return_value = [DummyProvider("sentence-transformers")]

    embedding_model = DummyModel("sentence-transformers/all-MiniLM-L6-v2", model_type="embedding")
    embedding_model.custom_metadata["embedding_dimension"] = 384
    embedding_model.custom_metadata["provider_id"] = "sentence-transformers"
    available_models.append(embedding_model)

    result = ogx_conftest.ogx_models.__wrapped__(ogx_client=mock_client)
    assert result.model_id == expected_model_id


def test_ogx_models_selection_no_llm_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ogx_conftest, "OGX_CORE_INFERENCE_MODEL", "")

    mock_client = MagicMock()
    embedding_model = DummyModel("sentence-transformers/all-MiniLM-L6-v2", model_type="embedding")
    embedding_model.custom_metadata["embedding_dimension"] = 384
    mock_client.models.list.return_value.data = [embedding_model]

    with pytest.raises(ValueError, match="No LLM models found in OGX client"):
        ogx_conftest.ogx_models.__wrapped__(ogx_client=mock_client)
