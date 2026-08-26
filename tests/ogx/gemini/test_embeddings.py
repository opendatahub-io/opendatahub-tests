"""Embedding tests for the remote::gemini provider.

Covers test cases TC-EMB-001 and TC-EMB-002 from the remote_gemini_provider
test plan (RHAISTRAT-1245).
"""

import math

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
            {"name": "test-gemini-embeddings", "randomize_name": True},
            {"enable_gemini": True},
            id="gemini",
        ),
    ],
    indirect=True,
)
class TestGeminiEmbeddings:
    """Embedding generation and graceful missing-usage handling for Gemini."""

    @pytest.mark.tier1
    def test_embedding_generation(
        self,
        ogx_client: OgxClient,
        gemini_embedding_model_id: str,
    ) -> None:
        """Verify embeddings via remote::gemini return valid vectors (TC-EMB-001).

        Given: an active remote::gemini provider with an embedding model.
        When: an embeddings request is sent for a sample text.
        Then: the response contains at least one embedding of non-zero, consistent
            dimension made up of floating-point numbers.
        """
        response = ogx_client.embeddings.create(
            model=gemini_embedding_model_id,
            input="Sample text for embedding generation",
        )
        assert response.data, "Embeddings response contains no data"

        embedding = response.data[0].embedding
        assert isinstance(embedding, list), f"Expected a list embedding, got {type(embedding).__name__}"
        assert len(embedding) > 0, "Embedding vector has zero dimension"
        assert all(isinstance(value, float) and math.isfinite(value) for value in embedding), (
            "Embedding vector contains non-float or non-finite values"
        )

    @pytest.mark.tier1
    def test_embedding_missing_usage_handled(
        self,
        ogx_client: OgxClient,
        gemini_embedding_model_id: str,
    ) -> None:
        """Verify embeddings succeed even when usage stats are omitted (TC-EMB-002).

        Given: an active remote::gemini provider whose API may omit usage stats.
        When: an embeddings request is sent.
        Then: the request succeeds with valid vectors and no AttributeError/HTTP 500,
            regardless of whether usage is present (and usage, if present, is valid).
        """
        response = ogx_client.embeddings.create(
            model=gemini_embedding_model_id,
            input="Test input for embedding with missing usage stats",
        )
        assert response.data, "Embeddings response contains no data"
        assert isinstance(response.data[0].embedding, list), "Embedding is not a list"
        assert len(response.data[0].embedding) > 0, "Embedding vector has zero dimension"

        usage = getattr(response, "usage", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            if prompt_tokens is not None:
                assert prompt_tokens >= 0, f"Invalid prompt_tokens in usage: {prompt_tokens!r}"
        else:
            LOGGER.info("Gemini embeddings response omitted usage statistics; request still succeeded")
