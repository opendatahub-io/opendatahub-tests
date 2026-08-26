"""Tool calling tests for the remote::gemini provider.

Covers test cases TC-TOOL-001 and TC-TOOL-002 from the remote_gemini_provider
test plan (RHAISTRAT-1245).
"""

import json

import pytest
import structlog
from ogx_client import OgxClient

LOGGER = structlog.get_logger(name=__name__)

# These tests require live Gemini API access and must not run on disconnected clusters.
pytestmark = [pytest.mark.skip_on_disconnected]

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}

# Tool definition intentionally packed with JSON schema fields Gemini does not
# support; the remote::gemini provider must strip these before calling Gemini.
UNSUPPORTED_SCHEMA_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_item",
        "description": "Look up an item by ID",
        "parameters": {
            "type": "object",
            "$schema": "http://json-schema.org/draft-07/schema#",
            "properties": {
                "item_id": {
                    "type": "integer",
                    "exclusiveMinimum": 0,
                    "exclusiveMaximum": 10000,
                    "default": 1,
                },
                "tags": {"type": "array", "minItems": 1, "maxItems": 10},
            },
            "required": ["item_id"],
            "additionalProperties": False,
        },
    },
}


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ogx_server",
    [
        pytest.param(
            {"name": "test-gemini-tools", "randomize_name": True},
            {"enable_gemini": True},
            id="gemini",
        ),
    ],
    indirect=True,
)
class TestGeminiToolCalling:
    """Tool calling behavior for the remote::gemini provider."""

    @pytest.mark.tier1
    def test_tool_calling(
        self,
        ogx_client: OgxClient,
        gemini_model_id: str,
    ) -> None:
        """Verify tool calling returns valid tool_calls (TC-TOOL-001).

        Given: an active remote::gemini provider and a get_weather tool.
        When: a prompt that should trigger the tool is sent with the tool defined.
        Then: the assistant message contains a well-formed tool_call for get_weather
            with parseable JSON arguments and finish_reason "tool_calls".
        """
        response = ogx_client.chat.completions.create(
            model=gemini_model_id,
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[WEATHER_TOOL],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )
        assert response.choices, "Tool-calling request returned no choices"
        choice = response.choices[0]
        tool_calls = choice.message.tool_calls
        assert tool_calls, "Expected tool_calls in the assistant message"

        tool_call = tool_calls[0]
        assert tool_call.type == "function", f"Expected type 'function', got {tool_call.type!r}"
        assert tool_call.id, "Tool call is missing an id"
        assert tool_call.function.name == "get_weather", f"Expected tool 'get_weather', got {tool_call.function.name!r}"
        arguments = json.loads(tool_call.function.arguments)
        assert isinstance(arguments, dict), "Tool call arguments did not parse to a JSON object"
        assert choice.finish_reason == "tool_calls", (
            f"Expected finish_reason 'tool_calls', got {choice.finish_reason!r}"
        )

    @pytest.mark.tier3
    def test_tool_calling_with_unsupported_schema_fields(
        self,
        ogx_client: OgxClient,
        gemini_model_id: str,
    ) -> None:
        """Verify unsupported JSON schema fields are filtered (TC-TOOL-002).

        Given: an active remote::gemini provider and a tool whose schema contains
            fields Gemini rejects (additionalProperties, $schema, exclusiveMinimum,
            exclusiveMaximum, maxItems, minItems, default, ...).
        When: a chat completion request with that tool is sent.
        Then: the request succeeds (HTTP 200) instead of failing with a Gemini
            schema-validation error, because the provider strips unsupported fields.
        """
        response = ogx_client.chat.completions.create(
            model=gemini_model_id,
            messages=[{"role": "user", "content": "Look up item 42"}],
            tools=[UNSUPPORTED_SCHEMA_TOOL],
        )
        # The key assertion is that the call did not raise a 400 from Gemini's
        # schema validation; a valid completion (with or without a tool call) is
        # produced.
        assert response.choices, "Expected a valid chat completion despite unsupported schema fields"
        message = response.choices[0].message
        assert message.tool_calls or message.content, (
            "Response had neither tool_calls nor content after schema filtering"
        )
