import pytest
from ogx_client import OgxClient

from tests.ogx.constants import ModelInfo


@pytest.mark.parametrize(
    "unprivileged_model_namespace",
    [
        pytest.param(
            {"name": "test-ogx-responses", "randomize_name": True},
        ),
    ],
    indirect=True,
)
@pytest.mark.rag
@pytest.mark.skip_must_gather
class TestOgxResponses:
    """Test class for OGX responses API functionality.

    For more information about this API, see:
    - https://github.com/ogx-ai/ogx-client-python/blob/main/api.md#responses
    - https://github.com/openai/openai-python/blob/main/api.md#responses
    """

    @pytest.mark.tier1
    def test_responses_create(
        self,
        ogx_client: OgxClient,
        ogx_models: ModelInfo,
    ) -> None:
        """
        Test simple responses API from the ogx server.

        Validates basic text generation capabilities using the responses API endpoint.
        Tests factual questions with constrained answers to ensure the LLM can
        provide correct responses across different model sizes.
        """
        test_cases = [
            ("What is the capital of France?", ["paris"]),
            ("What programming language is executed inside web browsers (client-side)?", ["javascript"]),
            ("Name a primary color in the RYB color model.", ["red", "yellow", "blue"]),
            ("What is 15 + 27?", ["42", "forty-two"]),
            ("Summarize what Python is in one sentence.", ["programming", "language"]),
        ]

        for question, expected_keywords in test_cases:
            response = ogx_client.responses.create(
                model=ogx_models.model_id,
                input=question,
                instructions="You are a helpful assistant.",
                temperature=0.0,
                max_output_tokens=4096,
            )

            content = response.output_text
            assert content is not None, "LLM response content is None"
            assert any(keyword in content.lower() for keyword in expected_keywords), (
                f"The LLM didn't provide any of the expected keywords {expected_keywords}. Got: {content}"
            )
