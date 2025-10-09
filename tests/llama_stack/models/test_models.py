import pytest

from tests.llama_stack.constants import LlamaStackProviders
from llama_stack_client import LlamaStackClient
from utilities.constants import MinIo, QWEN_MODEL_NAME
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret


@pytest.mark.parametrize(
    "model_namespace, minio_pod, minio_data_connection, llama_stack_server_config",
    [
        pytest.param(
            {"name": "test-llamastack-models"},
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
class TestLlamaStackModels:
    """Test class for LlamaStack models API functionality.

    For more information about this API, see:
    - https://github.com/llamastack/llama-stack-client-python/blob/main/api.md#models
    - https://github.com/openai/openai-python/blob/main/api.md#models
    """

    def test_models_list(
        self, minio_pod: Pod, minio_data_connection: Secret, llama_stack_client: LlamaStackClient
    ) -> None:
        """Test the initial state of the LlamaStack server and available models."""
        models = llama_stack_client.models.list()
        assert models is not None, "No models returned from LlamaStackClient"

        llm_model = next((m for m in models if m.api_model_type == "llm"), None)
        assert llm_model is not None, "No LLM model found in available models"
        model_id = llm_model.identifier
        assert model_id is not None, "No identifier set in LLM model"

        embedding_model = next((m for m in models if m.api_model_type == "embedding"), None)
        assert embedding_model is not None, "No embedding model found in available models"
        embedding_model_id = embedding_model.identifier
        assert embedding_model_id is not None, "No embedding model returned from LlamaStackClient"
        assert "embedding_dimension" in embedding_model.metadata, "embedding_dimension not found in model metadata"
        embedding_dimension = embedding_model.metadata["embedding_dimension"]
        assert embedding_dimension is not None, "No embedding_dimension set in embedding model"

    def test_models_register(
        self, minio_pod: Pod, minio_data_connection: Secret, llama_stack_client: LlamaStackClient
    ) -> None:
        """Test model registration functionality."""
        response = llama_stack_client.models.register(
            provider_id=LlamaStackProviders.Inference.VLLM_INFERENCE, model_type="llm", model_id=QWEN_MODEL_NAME
        )
        assert response

    def test_model_list(
        self, minio_pod: Pod, minio_data_connection: Secret, llama_stack_client: LlamaStackClient
    ) -> None:
        """Test listing available models and verify their properties."""
        models = llama_stack_client.models.list()

        # We only need to check the first model;
        # Second and third are embedding models present by default
        assert len(models) >= 2
        assert models[0].identifier == f"{LlamaStackProviders.Inference.VLLM_INFERENCE.value}/{QWEN_MODEL_NAME}"
        assert models[0].model_type == "llm"
        assert models[0].provider_id == LlamaStackProviders.Inference.VLLM_INFERENCE
