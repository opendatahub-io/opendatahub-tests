import pytest
from llama_stack_client import LlamaStackClient

from tests.llama_stack.constants import ModelInfo


def _assert_chat_completion_ack(
    unprivileged_llama_stack_client: LlamaStackClient,
    llama_stack_models: ModelInfo,
) -> None:
    response = unprivileged_llama_stack_client.chat.completions.create(
        model=llama_stack_models.model_id,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Just respond ACK."},
        ],
        temperature=0,
    )
    assert len(response.choices) > 0, "No response after basic inference on llama-stack server"

    content = response.choices[0].message.content
    assert content is not None, "LLM response content is None"
    assert "ack" in content.lower(), "The LLM did not provide the expected answer to the prompt"


@pytest.mark.parametrize(
    "unprivileged_model_namespace",
    [
        pytest.param(
            {"name": "test-llamastack-infer-chat-upgrade"},
        ),
    ],
    indirect=True,
)
@pytest.mark.llama_stack
class TestPreUpgradeLlamaStackInferenceCompletions:
    @pytest.mark.pre_upgrade
    def test_inference_chat_completion_pre_upgrade(
        self,
        unprivileged_llama_stack_client: LlamaStackClient,
        llama_stack_models: ModelInfo,
    ) -> None:
        """Verify chat completion returns ACK before upgrade.

        Given: A running unprivileged LlamaStack distribution.
        When: A deterministic chat completion request is sent.
        Then: The response contains at least one choice with non-empty ACK content.
        """
        _assert_chat_completion_ack(
            unprivileged_llama_stack_client=unprivileged_llama_stack_client,
            llama_stack_models=llama_stack_models,
        )


@pytest.mark.parametrize(
    "unprivileged_model_namespace",
    [
        pytest.param(
            {"name": "test-llamastack-infer-chat-upgrade"},
        ),
    ],
    indirect=True,
)
@pytest.mark.llama_stack
class TestPostUpgradeLlamaStackInferenceCompletions:
    @pytest.mark.post_upgrade
    @pytest.mark.xfail(reason="RHAIENG-3650")
    def test_inference_chat_completion_post_upgrade(
        self,
        unprivileged_llama_stack_client: LlamaStackClient,
        llama_stack_models: ModelInfo,
    ) -> None:
        """Verify chat completion returns ACK after upgrade.

        Given: A pre-existing unprivileged LlamaStack distribution after platform upgrade.
        When: A deterministic chat completion request is sent.
        Then: The response contains at least one choice with non-empty ACK content.
        """
        _assert_chat_completion_ack(
            unprivileged_llama_stack_client=unprivileged_llama_stack_client,
            llama_stack_models=llama_stack_models,
        )
