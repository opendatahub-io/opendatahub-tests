import pytest

from tests.llama_stack.constants import LlamaStackProviders


@pytest.mark.parametrize(
    "model_namespace, llama_stack_server_config",
    [
        pytest.param(
            {"name": "test-guardrails-lls"},
            {"vllm_url_fixture": "qwen25_05B_instruct_url", "inference_model": "qwen"},
        )
    ],
    indirect=True,
)
@pytest.mark.rawdeployment
class TestLlamaStackCore:
    def test_model_register(self, llama_stack_client):
        response = llama_stack_client.models.register(
            provider_id=LlamaStackProviders.Inference.VLLM_INFERENCE, model_type="llm", model_id="qwen"
        )
        assert response

    def test_model_list(self, llama_stack_client):
        models = llama_stack_client.models.list()

        # We only need to check the first model;
        # second is a granite embedding model present by default
        assert len(models) == 2
        assert models[0].identifier == "qwen"
        assert models[0].model_type == "llm"
        assert models[0].provider_id == LlamaStackProviders.Inference.VLLM_INFERENCE

    def test_inference(self, llama_stack_client):
        response = llama_stack_client.inference.chat_completion(
            messages=[
                {"role": "system", "content": "You are a friendly assistant."},
                {"role": "user", "content": "Only respond with ack"},
            ],
            model_id="qwen",
        )

        assert response.completion_message.content != ""
