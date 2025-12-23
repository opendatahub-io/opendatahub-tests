import pytest
import os
from tests.llama_stack.constants import LlamaStackProviders
from llama_stack_client import LlamaStackClient, NotFoundError
from llama_stack_client.types import Model


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

        llm_model = next((model for model in models if model.api_model_type == "llm"), None)
        assert llm_model is not None, "No LLM model found in available models"
        assert isinstance(llm_model, Model), "LLM model should be a Model instance"
        assert llm_model.identifier is not None, "No identifier set in LLM model"
        assert len(llm_model.identifier) > 0, "LLM model identifier should not be empty"

        embedding_model = next((model for model in models if model.api_model_type == "embedding"), None)
        assert embedding_model is not None, "No embedding model found in available models"
        assert isinstance(embedding_model, Model), "Embedding model should be a Model instance"
        assert embedding_model.identifier is not None, "No identifier set in embedding model"
        assert len(embedding_model.identifier) > 0, "Embedding model identifier should not be empty"
        assert "embedding_dimension" in embedding_model.metadata, "embedding_dimension not found in model metadata"
        embedding_dimension = embedding_model.metadata["embedding_dimension"]
        assert embedding_dimension is not None, "No embedding_dimension set in embedding model"
        # API returns dimension as float (e.g., 768.0) though conceptually an integer
        assert isinstance(embedding_dimension, float), "embedding_dimension should be a float"
        assert embedding_dimension > 0, "embedding_dimension should be positive"
        assert embedding_dimension.is_integer(), "embedding_dimension should be a whole number"

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
            assert hasattr(model, "identifier"), "Model should have identifier attribute"
            assert hasattr(model, "api_model_type"), "Model should have api_model_type attribute"
            assert model.identifier is not None, f"Model {model} should have a non-None identifier"
            assert model.api_model_type in ["llm", "embedding"], (
                f"Model {model.identifier} should have api_model_type 'llm' or 'embedding', "
                f"got '{model.api_model_type}'"
            )
            assert hasattr(model, "metadata"), "Model should have metadata attribute"
            assert isinstance(model.metadata, dict), "Model metadata should be a dictionary"

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
        retrieved_model = unprivileged_llama_stack_client.models.retrieve(model_id=test_model.identifier)

        assert retrieved_model is not None, f"Model {test_model.identifier} should be retrievable"
        assert isinstance(retrieved_model, Model), "Retrieved model should be a Model instance"
        assert retrieved_model.identifier == test_model.identifier, (
            f"Retrieved model identifier '{retrieved_model.identifier}' "
            f"should match requested '{test_model.identifier}'"
        )
        assert retrieved_model.api_model_type == test_model.api_model_type, (
            f"Retrieved model type '{retrieved_model.api_model_type}' "
            f"should match original '{test_model.api_model_type}'"
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

    def test_models_register(
        self,
        unprivileged_llama_stack_client: LlamaStackClient,
    ) -> None:
        """Test registering a new model.

        Verifies that models.register() successfully registers a new model
        and it appears in the models list.
        """
        inference_model = os.getenv("LLS_CORE_INFERENCE_MODEL")
        assert inference_model, "LLS_CORE_INFERENCE_MODEL environment variable must be set"
        test_model_id = f"{inference_model}-test-register"

        response = unprivileged_llama_stack_client.models.register(
            model_id=test_model_id,
            model_type="llm",
            provider_id=LlamaStackProviders.Inference.VLLM_INFERENCE,
        )
        assert response is not None, "Model registration should return a response"

        registered_model_id = f"{LlamaStackProviders.Inference.VLLM_INFERENCE.value}/{test_model_id}"
        try:
            models = unprivileged_llama_stack_client.models.list()
            registered_model_ids = [model.identifier for model in models]
            assert registered_model_id in registered_model_ids, (
                f"Registered model {registered_model_id} should appear in models list"
            )
        finally:
            unprivileged_llama_stack_client.models.unregister(model_id=registered_model_id)

    def test_models_register_retrieve_unregister(
        self,
        unprivileged_llama_stack_client: LlamaStackClient,
    ) -> None:
        """Test complete model lifecycle: register, retrieve, and unregister.

        Verifies the full workflow of registering a model, retrieving it,
        verifying its properties, and then unregistering it.
        """
        inference_model = os.getenv("LLS_CORE_INFERENCE_MODEL")
        assert inference_model, "LLS_CORE_INFERENCE_MODEL environment variable must be set"
        test_model_id = f"{inference_model}-test-lifecycle"

        response = unprivileged_llama_stack_client.models.register(
            model_id=test_model_id,
            model_type="llm",
            provider_id=LlamaStackProviders.Inference.VLLM_INFERENCE,
        )
        assert response is not None, "Model registration should return a response"

        registered_model_id = f"{LlamaStackProviders.Inference.VLLM_INFERENCE.value}/{test_model_id}"
        try:
            registered_model = unprivileged_llama_stack_client.models.retrieve(model_id=registered_model_id)
            assert registered_model is not None, f"LLM {registered_model_id} not found using models.retrieve"
            assert isinstance(registered_model, Model), "Retrieved model should be a Model instance"
            expected_id_suffix = f"/{test_model_id}"
            assert registered_model.identifier.endswith(expected_id_suffix), (
                f"Model identifier '{registered_model.identifier}' should end with '{expected_id_suffix}'"
            )
            assert registered_model.api_model_type == "llm", (
                f"Registered model should have api_model_type 'llm', got '{registered_model.api_model_type}'"
            )
            assert registered_model.provider_id == LlamaStackProviders.Inference.VLLM_INFERENCE.value, (
                f"Registered model provider_id should be '{LlamaStackProviders.Inference.VLLM_INFERENCE.value}', "
                f"got '{registered_model.provider_id}'"
            )
        finally:
            unprivileged_llama_stack_client.models.unregister(model_id=registered_model_id)

        with pytest.raises(NotFoundError):
            unprivileged_llama_stack_client.models.retrieve(model_id=registered_model_id)
