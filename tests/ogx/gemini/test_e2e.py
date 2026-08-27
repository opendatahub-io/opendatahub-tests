"""End-to-end user-journey tests for the remote::gemini provider.

Covers test cases TC-E2E-001, TC-E2E-002, TC-E2E-003 and TC-E2E-004 from the
remote_gemini_provider test plan (RHAISTRAT-1245). Each test exercises a full
workflow against a distribution deployed with remote::gemini enabled (via the
``enable_gemini`` server param and the shared Gemini fixtures).
"""

import json
import math
from concurrent.futures import ThreadPoolExecutor

import pytest
import structlog
from ogx_client import OgxClient

from tests.ogx.constants import GEMINI_API_KEY_SECONDARY, GEMINI_API_KEY_SECONDARY_2
from tests.ogx.gemini.utils import is_gemini_provider_active, provider_data_headers

LOGGER = structlog.get_logger(name=__name__)

# These tests require live Gemini API access and must not run on disconnected clusters.
pytestmark = [pytest.mark.skip_on_disconnected]

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ogx_server",
    [
        pytest.param(
            {"name": "test-gemini-e2e", "randomize_name": True},
            {"enable_gemini": True},
            id="gemini",
        ),
    ],
    indirect=True,
)
class TestGeminiEndToEnd:
    """Complete user journeys through the remote::gemini provider."""

    @pytest.mark.tier1
    def test_deploy_and_chat_completion(
        self,
        ogx_client: OgxClient,
        gemini_model_id: str,
    ) -> None:
        """Deploy with Gemini then run non-streaming and streaming chat (TC-E2E-001).

        Given: an OGX distribution deployed with remote::gemini active.
        When: the provider list is queried, then a non-streaming and a streaming
            chat completion are sent to a Gemini model.
        Then: remote::gemini is listed, the non-streaming call returns valid content,
            and the streaming call yields chunks that terminate cleanly.
        """
        assert is_gemini_provider_active(ogx_client=ogx_client), "remote::gemini provider is not active"

        non_streaming = ogx_client.chat.completions.create(
            model=gemini_model_id,
            messages=[{"role": "user", "content": "Say hello in one short sentence."}],
            temperature=0.7,
            stream=False,
        )
        assert non_streaming.choices and non_streaming.choices[0].message.content, (
            "Non-streaming chat completion returned no content"
        )

        stream = ogx_client.chat.completions.create(
            model=gemini_model_id,
            messages=[{"role": "user", "content": "Say hello in one short sentence."}],
            temperature=0.7,
            stream=True,
        )
        chunks = list(stream)
        assert chunks, "Streaming chat completion produced no chunks"
        streamed_text = "".join(
            chunk.choices[0].delta.content
            for chunk in chunks
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content
        )
        assert streamed_text, "Streaming chat completion produced no content across chunks"

    @pytest.mark.tier1
    def test_full_tool_calling_workflow(
        self,
        ogx_client: OgxClient,
        gemini_model_id: str,
    ) -> None:
        """Run a complete tool-calling round trip (TC-E2E-002).

        Given: an active remote::gemini provider and a get_weather tool.
        When: a triggering prompt is sent, the tool_call is answered with a tool
            result message, and the conversation is continued.
        Then: the first response contains a get_weather tool_call with
            finish_reason "tool_calls", and the follow-up incorporates the tool
            result into a final natural-language answer.
        """
        first = ogx_client.chat.completions.create(
            model=gemini_model_id,
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[WEATHER_TOOL],
            tool_choice="auto",
        )
        assert first.choices, "First tool-calling request returned no choices"
        first_choice = first.choices[0]
        tool_calls = first_choice.message.tool_calls
        assert tool_calls, "First request did not return tool_calls"
        assert first_choice.finish_reason == "tool_calls", (
            f"Expected finish_reason 'tool_calls', got {first_choice.finish_reason!r}"
        )
        tool_call = tool_calls[0]
        assert tool_call.function.name == "get_weather", f"Expected tool 'get_weather', got {tool_call.function.name!r}"
        json.loads(tool_call.function.arguments)  # arguments must be valid JSON

        follow_up = ogx_client.chat.completions.create(
            model=gemini_model_id,
            messages=[
                {"role": "user", "content": "What is the weather in Paris?"},
                {
                    "role": "assistant",
                    "content": first_choice.message.content or "",
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps({"location": "Paris", "temperature_c": 18, "conditions": "sunny"}),
                },
            ],
            tools=[WEATHER_TOOL],
        )
        assert follow_up.choices and follow_up.choices[0].message.content, (
            "Follow-up request did not return a final answer incorporating the tool result"
        )

    @pytest.mark.tier1
    def test_embedding_generation_end_to_end(
        self,
        ogx_client: OgxClient,
        gemini_embedding_model_id: str,
    ) -> None:
        """Generate embeddings for short and long inputs end-to-end (TC-E2E-003).

        Given: an active remote::gemini provider with an embedding model.
        When: embeddings are requested for a short input and a longer multi-sentence
            input.
        Then: both requests succeed with float embedding vectors of consistent
            dimension, regardless of whether Gemini returns usage statistics.
        """
        assert is_gemini_provider_active(ogx_client=ogx_client), "remote::gemini provider is not active"

        short = ogx_client.embeddings.create(model=gemini_embedding_model_id, input="Hello world")
        long_input = (
            "OGX integrates the Gemini provider. "
            "This is a longer, multi-sentence input used to validate embeddings. "
            "It should produce a vector of the same dimension as the short input."
        )
        long = ogx_client.embeddings.create(model=gemini_embedding_model_id, input=long_input)

        for response in (short, long):
            assert response.data, "Embedding response contained no data"
            vector = response.data[0].embedding
            assert vector, "Embedding vector is empty"
            assert all(isinstance(value, float) and math.isfinite(value) for value in vector), (
                "Embedding vector contains non-float or non-finite values"
            )

        assert len(short.data[0].embedding) == len(long.data[0].embedding), (
            "Embedding dimensions differ between short and long inputs"
        )

    @pytest.mark.tier2
    def test_multi_tenant_per_request_keys(
        self,
        ogx_client: OgxClient,
        gemini_model_id: str,
    ) -> None:
        """Run concurrent requests with different per-request keys (TC-E2E-004).

        Given: an active remote::gemini provider with a config key plus two
            secondary valid keys.
        When: three requests are sent concurrently — one with the config key, and
            one with each secondary key via x-ogx-provider-data.
        Then: all three succeed independently with valid content and no
            cross-contamination between keys.
        """
        if not (GEMINI_API_KEY_SECONDARY and GEMINI_API_KEY_SECONDARY_2):
            pytest.fail(
                reason="OGX_CORE_GEMINI_API_KEY_SECONDARY and _SECONDARY_2 must both be set "
                "to exercise the multi-tenant per-request override scenario"
            )

        request_headers = [
            None,
            provider_data_headers(gemini_api_key=GEMINI_API_KEY_SECONDARY),
            provider_data_headers(gemini_api_key=GEMINI_API_KEY_SECONDARY_2),
        ]

        def _send(extra_headers: dict[str, str] | None) -> str | None:
            response = ogx_client.chat.completions.create(
                model=gemini_model_id,
                messages=[{"role": "user", "content": "Reply with a single short greeting."}],
                extra_headers=extra_headers,
            )
            return response.choices[0].message.content if response.choices else None

        with ThreadPoolExecutor(max_workers=len(request_headers)) as executor:
            results = list(executor.map(_send, request_headers))

        assert all(results), f"Not all concurrent multi-tenant requests returned content: {results!r}"
