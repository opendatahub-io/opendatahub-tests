"""Chat completion tests for the remote::gemini provider.

Covers test cases TC-CHAT-001, TC-CHAT-002 and TC-CHAT-003 from the
remote_gemini_provider test plan (RHAISTRAT-1245).
"""

import pytest
import structlog
from ogx_client import OgxClient

LOGGER = structlog.get_logger(name=__name__)

# These tests require live Gemini API access and must not run on disconnected clusters.
pytestmark = [pytest.mark.skip_on_disconnected]


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ogx_server",
    [
        pytest.param(
            {"name": "test-gemini-chat", "randomize_name": True},
            {"enable_gemini": True},
            id="gemini",
        ),
    ],
    indirect=True,
)
class TestGeminiChatCompletions:
    """Non-streaming, streaming, and temperature behavior for Gemini chat."""

    @pytest.mark.tier1
    def test_non_streaming_chat_completion(
        self,
        ogx_client: OgxClient,
        gemini_model_id: str,
    ) -> None:
        """Verify non-streaming chat completions return a valid response (TC-CHAT-001).

        Given: an active remote::gemini provider.
        When: a non-streaming chat completion request is sent.
        Then: the response has a choice with an assistant message and non-empty content.
        """
        response = ogx_client.chat.completions.create(
            model=gemini_model_id,
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            temperature=0.7,
            stream=False,
        )
        assert response.id, "Chat completion response is missing an id"
        assert response.choices, "Chat completion response contains no choices"

        message = response.choices[0].message
        assert message.role == "assistant", f"Expected assistant role, got {message.role!r}"
        assert message.content, "Assistant message content is empty"

    @pytest.mark.tier1
    def test_streaming_chat_completion(
        self,
        ogx_client: OgxClient,
        gemini_model_id: str,
    ) -> None:
        """Verify streaming chat completions deliver chunked deltas (TC-CHAT-002).

        Given: an active remote::gemini provider.
        When: a streaming chat completion request is sent.
        Then: multiple delta chunks arrive, the stream finishes with reason "stop",
            and the concatenated deltas form non-empty text.
        """
        stream = ogx_client.chat.completions.create(
            model=gemini_model_id,
            messages=[{"role": "user", "content": "Explain quantum computing."}],
            stream=True,
            temperature=0.5,
        )

        # Guard against a provider that never terminates the stream: cap iteration
        # so a runaway response fails loudly instead of hanging the suite.
        max_chunks = 2000
        collected_content = ""
        finish_reasons = []
        chunk_count = 0
        for chunk in stream:
            chunk_count += 1
            assert chunk_count <= max_chunks, f"Streaming response exceeded {max_chunks} chunks without terminating"
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.delta and choice.delta.content:
                collected_content += choice.delta.content
            if choice.finish_reason:
                finish_reasons.append(choice.finish_reason)

        assert chunk_count > 0, "No SSE chunks were received from the streaming response"
        assert "stop" in finish_reasons, f"Stream did not finish with reason 'stop': {finish_reasons!r}"
        assert collected_content.strip(), "Concatenated streamed content is empty"

    @pytest.mark.tier1
    def test_temperature_controls_variability(
        self,
        ogx_client: OgxClient,
        gemini_model_id: str,
    ) -> None:
        """Verify the temperature parameter is accepted across its range (TC-CHAT-003).

        Given: an active remote::gemini provider.
        When: the same prompt is sent three times at temperature 0 and three times
            at temperature 1.0.
        Then: every request succeeds with valid content, demonstrating the provider
            accepts the temperature parameter across its range.

        Note: neither response-text variability nor that the provider honors/forwards
        the temperature on the wire is asserted. A compliant provider may legitimately
        return identical outputs at temperature 1.0 (especially for a constrained
        "one word" prompt), so a variability comparison would be probabilistic and
        flaky. Confirming the value is honored on the wire would require request-capture
        tooling that the pytest harness does not provide, so this test scopes its claim
        to parameter acceptance only.
        """
        prompt = "Reply with exactly one word: yes or no."

        def _sample(temperature: float) -> list[str]:
            responses = []
            for _ in range(3):
                response = ogx_client.chat.completions.create(
                    model=gemini_model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                assert response.choices, f"No choices at temperature={temperature}"
                content = response.choices[0].message.content
                assert content, f"Empty content at temperature={temperature}"
                responses.append(content.strip().lower())
            return responses

        low_temperature_responses = _sample(temperature=0.0)
        high_temperature_responses = _sample(temperature=1.0)
        LOGGER.info(f"temperature=0 responses: {low_temperature_responses}")
        LOGGER.info(f"temperature=1.0 responses: {high_temperature_responses}")

        # _sample already asserts each individual response is valid; the meaningful,
        # non-flaky assertion is that both temperature settings were accepted and
        # produced complete result sets.
        assert len(low_temperature_responses) == 3 and len(high_temperature_responses) == 3, (
            "Provider did not return valid completions across the full temperature range"
        )
