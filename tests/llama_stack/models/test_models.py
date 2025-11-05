import pytest

from tests.llama_stack.constants import LlamaStackProviders
from llama_stack_client import LlamaStackClient
from utilities.constants import MinIo, QWEN_MODEL_NAME


@pytest.mark.parametrize(
    "model_namespace, minio_pod, minio_data_connection, llama_stack_server_config",
    [
        pytest.param(
            {"name": "test-llamastack-models", "randomize_name": True},
            MinIo.PodConfig.QWEN_HAP_BPIV2_MINIO_CONFIG,
            {"bucket": "llms"},
            {
                "vllm_url_fixture": "qwen_isvc_url",
                "inference_model": QWEN_MODEL_NAME,
                "llama_stack_storage_size": "2Gi",
            },
        )
    ],
    indirect=True,
)
@pytest.mark.rawdeployment
@pytest.mark.smoke
@pytest.mark.llama_stack
@pytest.mark.skip_must_gather
@pytest.mark.usefixtures("minio_pod", "minio_data_connection")
class TestLlamaStackModels:
    """Test class for LlamaStack models API functionality.

    For more information about this API, see:
    - https://github.com/llamastack/llama-stack-client-python/blob/main/api.md#models
    - https://github.com/openai/openai-python/blob/main/api.md#models
    """

    def test_models_list(self, llama_stack_client: LlamaStackClient) -> None:
        """Test the initial state of the LlamaStack server and available models."""
        models = llama_stack_client.models.list()
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

    def test_models_register(self, llama_stack_client: LlamaStackClient) -> None:
        """Test model registration functionality."""
        response = llama_stack_client.models.register(
            provider_id=LlamaStackProviders.Inference.VLLM_INFERENCE, model_type="llm", model_id=QWEN_MODEL_NAME
        )
        assert response

    def test_model_list(self, llama_stack_client: LlamaStackClient) -> None:
        """Test listing available models and verify their properties."""
        models = llama_stack_client.models.list()

        # Find the registered LLM by identifier suffix
        expected_id_suffix = f"/{QWEN_MODEL_NAME}"
        target = next(
            (model for model in models if model.model_type == "llm" and model.identifier.endswith(expected_id_suffix)),
            None,
        )
        assert target is not None, (
            f"LLM {QWEN_MODEL_NAME} not found in models: {[model.identifier for model in models]}"
        )
        assert target.identifier.endswith(expected_id_suffix)
        assert target.model_type == "llm"
        assert target.provider_id == LlamaStackProviders.Inference.VLLM_INFERENCE.value
