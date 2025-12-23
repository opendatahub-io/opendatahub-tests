import pytest
import os

from tests.llama_stack.constants import LlamaStackProviders
from llama_stack_client import LlamaStackClient, NotFoundError


@pytest.mark.parametrize(
    "unprivileged_model_namespace",
    [
        pytest.param(
            {"name": "test-llamastack-models", "randomize_name": True},
        ),
    ],
    indirect=True,
)
@pytest.mark.llama_stack
@pytest.mark.smoke
class TestLlamaStackModels:
    """Test class for LlamaStack models API functionality.

    For more information about this API, see:
    - https://github.com/llamastack/llama-stack-client-python/blob/main/api.md#models
    - https://github.com/openai/openai-python/blob/main/api.md#models
    """

    @pytest.mark.smoke
    def test_models_list(
        self,
        unprivileged_llama_stack_client: LlamaStackClient,
    ) -> None:
        """Test the initial state of the LlamaStack server and available models."""
        models = unprivileged_llama_stack_client.models.list()
        assert models is not None, "No models returned from LlamaStackClient"

        llm_model = next((model for model in models if model.api_model_type == "llm"), None)
        assert llm_model is not None, "No LLM model found in available models"
        model_id = llm_model.identifier
        assert model_id is not None, "No identifier set in LLM model"

        embedding_model = next((model for model in models if model.api_model_type == "embedding"), None)
        assert embedding_model is not None, "No embedding model found in available models"
        embedding_model_id = embedding_model.identifier
        assert embedding_model_id is not None, "No embedding model returned from LlamaStackClient"
        assert "embedding_dimension" in embedding_model.metadata, "embedding_dimension not found in model metadata"
        embedding_dimension = embedding_model.metadata["embedding_dimension"]
        assert embedding_dimension is not None, "No embedding_dimension set in embedding model"

    def test_models_register_retrieve_unregister(self, unprivileged_llama_stack_client: LlamaStackClient) -> None:
        """Test model registration functionality."""

        # Test model registration
        inference_model = os.getenv("LLS_CORE_INFERENCE_MODEL", "")
        test_model_id = f"{inference_model}-2"

        response = unprivileged_llama_stack_client.models.register(
            model_id=test_model_id,
            model_type="llm",
            provider_id=LlamaStackProviders.Inference.VLLM_INFERENCE,
        )
        assert response

        models = unprivileged_llama_stack_client.models.list()
        print(models)

        # Test model retrieval
        registered_model_id = f"{LlamaStackProviders.Inference.VLLM_INFERENCE.value}/{inference_model}-2"
        registered_model = unprivileged_llama_stack_client.models.retrieve(model_id=registered_model_id)
        assert registered_model is not None, f"LLM {registered_model_id} not found using models.retrieve"
        expected_id_suffix = f"/{test_model_id}"
        assert registered_model.identifier.endswith(expected_id_suffix)
        assert registered_model.api_model_type == "llm"
        assert registered_model.provider_id == LlamaStackProviders.Inference.VLLM_INFERENCE.value

        # Test model deletion
        unprivileged_llama_stack_client.models.unregister(model_id=registered_model_id)

        with pytest.raises(NotFoundError):
            unprivileged_llama_stack_client.models.retrieve(model_id=registered_model_id)
