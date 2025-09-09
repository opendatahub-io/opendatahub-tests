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
            {"name": "test-llamastack-core"},
            MinIo.PodConfig.QWEN_HAP_BPIV2_MINIO_CONFIG,
            {"bucket": "llms"},
            {"vllm_url_fixture": "qwen_isvc_url", "inference_model": QWEN_MODEL_NAME},
        )
    ],
    indirect=True,
)
@pytest.mark.rawdeployment
@pytest.mark.smoke
class TestLlamaStackCore:
    def test_lls_server_initial_state(
        self, minio_pod: Pod, minio_data_connection: Secret, llama_stack_client: LlamaStackClient
    ) -> None:
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

    def test_model_register(
        self, minio_pod: Pod, minio_data_connection: Secret, llama_stack_client: LlamaStackClient
    ) -> None:
        response = llama_stack_client.models.register(
            provider_id=LlamaStackProviders.Inference.VLLM_INFERENCE, model_type="llm", model_id=QWEN_MODEL_NAME
        )
        assert response

    def test_model_list(
        self, minio_pod: Pod, minio_data_connection: Secret, llama_stack_client: LlamaStackClient
    ) -> None:
        models = llama_stack_client.models.list()

        # We only need to check the first model;
        # Second and third are embedding models present by default
        assert len(models) >= 2
        assert models[0].identifier == f"{LlamaStackProviders.Inference.VLLM_INFERENCE.value}/{QWEN_MODEL_NAME}"
        assert models[0].model_type == "llm"
        assert models[0].provider_id == LlamaStackProviders.Inference.VLLM_INFERENCE

    def test_inference(
        self, minio_pod: Pod, minio_data_connection: Secret, llama_stack_client: LlamaStackClient
    ) -> None:
        response = llama_stack_client.chat.completions.create(
            model=QWEN_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Just respond ACK."},
            ],
            temperature=0,
        )

        assert len(response.choices) > 0, "No response after basic inference on llama-stack server"

        # Check if response has the expected structure and content
        content = response.choices[0].message.content
        assert content is not None, "LLM response content is None"
        assert "ACK" in content, "The LLM didn't provide the expected answer to the prompt"
