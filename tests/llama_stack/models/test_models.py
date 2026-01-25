import pytest
from llama_stack_client import LlamaStackClient, NotFoundError
from llama_stack_client.types import Model, ModelRetrieveResponse


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

    def test_models_list(
        self,
        unprivileged_llama_stack_client: LlamaStackClient,
    ) -> None:
        """Test listing all available models.

        Verifies that the models.list() method returns a non-empty list
        containing at least one LLM and one embedding model, compatible
        with OpenAI SDK structure.
        """
        models = unprivileged_llama_stack_client.models.list()
        assert models is not None, "No models returned from LlamaStackClient"
        assert isinstance(models, list), "models.list() should return a list"
        assert len(models) > 0, "At least one model should be available"

        llm_model = next((model for model in models if model.custom_metadata["model_type"] == "llm"), None)
        assert llm_model is not None, "No LLM model found in available models"
        assert isinstance(llm_model, Model), "LLM model should be a Model instance"
        assert llm_model.id is not None, "No identifier set in LLM model"
        assert len(llm_model.id) > 0, "LLM model identifier should not be empty"

        embedding_model = next((model for model in models if model.custom_metadata["model_type"] == "embedding"), None)
        assert embedding_model is not None, "No embedding model found in available models"
        assert isinstance(embedding_model, Model), "Embedding model should be a Model instance"
        assert embedding_model.id is not None, "No identifier set in embedding model"
        assert len(embedding_model.id) > 0, "Embedding model identifier should not be empty"
        assert "embedding_dimension" in embedding_model.custom_metadata, (
            "embedding_dimension not found in custom_metadata"
        )
        embedding_dimension = embedding_model.custom_metadata["embedding_dimension"]
        assert embedding_dimension is not None, "No embedding_dimension set in embedding model"
        # API returns dimension as integer (e.g., 768)
        assert isinstance(embedding_dimension, int), "embedding_dimension should be an integer"
        assert embedding_dimension > 0, "embedding_dimension should be positive"

    def test_models_list_structure(
        self,
        unprivileged_llama_stack_client: LlamaStackClient,
    ) -> None:
        """Test that model list response structure matches OpenAI SDK compatibility.

        Verifies that each model in the list has the required fields expected
        by OpenAI-compatible clients.
        """
        models = unprivileged_llama_stack_client.models.list()
        assert models is not None, "No models returned from LlamaStackClient"

        for model in models:
            assert hasattr(model, "id"), "Model should have identifier attribute"
            assert hasattr(model, "custom_metadata"), "Model should have custom_metadata attribute"
            assert isinstance(model.custom_metadata, dict), "Model custom_metadata should be a dictionary"
            assert model.id is not None, f"Model {model} should have a non-None identifier"
            assert model.custom_metadata["model_type"] in ["llm", "embedding"], (
                f"Model {model.id} should have custom_metadata[\"model_type\"] 'llm' or 'embedding', "
                f"got '{model.custom_metadata['model_type']}'"
            )

    def test_models_retrieve_existing(
        self,
        unprivileged_llama_stack_client: LlamaStackClient,
    ) -> None:
        """Test retrieving an existing model by ID.

        Verifies that models.retrieve() returns the correct model when given
        a valid model identifier from the list.
        """
        models = unprivileged_llama_stack_client.models.list()
        assert len(models) > 0, "At least one model should be available"

        test_model = models[0]
        retrieved_model = unprivileged_llama_stack_client.models.retrieve(model_id=test_model.id)

        assert retrieved_model is not None, f"Model {test_model.id} should be retrievable"
        assert isinstance(retrieved_model, ModelRetrieveResponse), "Retrieved model should be a ModelRetrieveResponse"
        assert retrieved_model.identifier == test_model.id, (
            f"Retrieved model identifier '{retrieved_model.identifier}' should match requested '{test_model.id}'"
        )
        assert retrieved_model.api_model_type == test_model.custom_metadata["model_type"], (
            f"Retrieved model type '{retrieved_model.api_model_type}' "
            f"should match original '{test_model.custom_metadata['model_type']}'"
        )

    def test_models_retrieve_nonexistent(
        self,
        unprivileged_llama_stack_client: LlamaStackClient,
    ) -> None:
        """Test retrieving a non-existent model raises NotFoundError.

        Verifies that models.retrieve() raises NotFoundError when given
        an invalid model identifier, matching OpenAI SDK behavior.
        """
        nonexistent_model_id = "nonexistent-provider/nonexistent-model"

        with pytest.raises(NotFoundError):
            unprivileged_llama_stack_client.models.retrieve(model_id=nonexistent_model_id)
