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
        unprivileged_ogx_client: OgxClient,
        ogx_models: ModelInfo,
    ) -> None:
        """
        Test simple responses API from the ogx server.

        Validates basic text generation capabilities using the responses API endpoint.
        Tests identity and capability questions to ensure the LLM can provide
        appropriate responses about itself and its functionality.
        """
        test_cases = [
            ("Who are you?", ["model", "assistant", "ai", "artificial", "language model"]),
            ("What can you do?", ["answer"]),
        ]

        for question, expected_keywords in test_cases:
            response = unprivileged_ogx_client.responses.create(
                model=ogx_models.model_id,
                input=question,
                instructions="You are a helpful assistant.",
            )

            content = response.output_text
            assert content is not None, "LLM response content is None"
            assert any(keyword in content.lower() for keyword in expected_keywords), (
                f"The LLM didn't provide any of the expected keywords {expected_keywords}. Got: {content}"
            )
