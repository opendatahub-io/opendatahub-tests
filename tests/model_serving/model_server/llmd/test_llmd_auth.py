import pytest

from tests.model_serving.model_server.llmd.utils import (
    ns_from_file,
    parse_completion_text,
    send_chat_completions,
)

pytestmark = [pytest.mark.tier1]

NAMESPACE = ns_from_file(file=__file__)


@pytest.mark.parametrize(
    "unprivileged_model_namespace",
    [{"name": NAMESPACE}],
    indirect=True,
)
class TestLLMISVCAuth:
    """Deploy TinyLlama on CPU with authentication enabled and verify access control on chat completions."""

    def test_llmisvc_authorized(self, llmisvc_auth_pair):
        """Test steps:

        1. Send a chat completion request to each service using its owner's token.
        2. Assert both responses return status 200.
        3. Assert both completion texts contain the expected answer.
        """
        entry_a, entry_b = llmisvc_auth_pair

        prompt = "What is the capital of Italy?"
        expected = "rome"

        for entry in [entry_a, entry_b]:
            status, body = send_chat_completions(
                llmisvc=entry.service,
                prompt=prompt,
                token=entry.token,
                insecure=False,
            )
            assert status == 200, f"Authorized request failed with {status}: {body}"
            completion = parse_completion_text(response_body=body)
            assert expected in completion.lower(), f"Expected '{expected}' in response, got: {completion}"

    def test_llmisvc_unauthorized(self, llmisvc_auth_pair):
        """Test steps:

        1. Send a chat completion request to user A's service using user B's token.
        2. Assert the response status is 401 or 403.
        3. Send a chat completion request to user A's service with no token.
        4. Assert the response status is 401 or 403.
        """
        entry_a, entry_b = llmisvc_auth_pair

        # User B's token cannot access user A's service
        status, _ = send_chat_completions(
            llmisvc=entry_a.service,
            prompt="What is the capital of Italy?",
            token=entry_b.token,
            insecure=False,
        )
        assert status in (401, 403), f"Cross-user access should be denied, got {status}"

        # No token at all fails
        status, _ = send_chat_completions(
            llmisvc=entry_a.service,
            prompt="What is the capital of Italy?",
            insecure=False,
        )
        assert status in (401, 403), f"No-token access should be denied, got {status}"
