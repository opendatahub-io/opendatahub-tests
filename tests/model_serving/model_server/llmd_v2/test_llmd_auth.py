"""Authentication tests for LLMInferenceService."""

import pytest

from tests.model_serving.model_server.llmd_v2.utils import (
    ns_from_file,
    parse_completion_text,
    send_chat_completions,
)

pytestmark = [pytest.mark.tier1, pytest.mark.cpu]

NAMESPACE = ns_from_file(__file__)


@pytest.mark.parametrize(
    "unprivileged_model_namespace",
    [{"name": NAMESPACE}],
    indirect=True,
)
class TestLLMISVCAuth:
    """Authentication testing for LLMD."""

    def test_llmisvc_authorized(self, llmisvc_auth_pair):
        """Test that authorized users can access their own LLMInferenceServices."""
        entry_a, entry_b = llmisvc_auth_pair

        prompt = "What is the capital of Italy?"
        expected = "rome"

        for entry in [entry_a, entry_b]:
            status, body = send_chat_completions(entry.service, prompt=prompt, token=entry.token, insecure=False)
            assert status == 200, f"Authorized request failed with {status}: {body}"
            completion = parse_completion_text(body)
            assert expected in completion.lower(), f"Expected '{expected}' in response, got: {completion}"

    def test_llmisvc_unauthorized(self, llmisvc_auth_pair):
        """Test that unauthorized access to LLMInferenceServices is properly blocked."""
        entry_a, entry_b = llmisvc_auth_pair

        # User B's token cannot access user A's service
        status, _ = send_chat_completions(
            entry_a.service, prompt="What is the capital of Italy?", token=entry_b.token, insecure=False
        )
        assert status in (401, 403), f"Cross-user access should be denied, got {status}"

        # No token at all fails
        status, _ = send_chat_completions(entry_a.service, prompt="What is the capital of Italy?", insecure=False)
        assert status in (401, 403), f"No-token access should be denied, got {status}"
