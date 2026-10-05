"""Regression tests ensuring remote::gemini does not disturb other providers.

Covers test cases TC-REG-001 and TC-REG-002 from the remote_gemini_provider
test plan (RHAISTRAT-1245).
"""

import pytest
import structlog
from ogx_client import OgxClient

from tests.ogx.constants import GEMINI_PROVIDER_ID, GEMINI_PROVIDER_TYPE
from tests.ogx.gemini.utils import list_provider_types

LOGGER = structlog.get_logger(name=__name__)

# These tests require live Gemini API access and must not run on disconnected clusters.
pytestmark = [pytest.mark.skip_on_disconnected]


def _resolve_model_id(ogx_client: OgxClient, provider_id: str, model_type: str) -> str | None:
    """Return the first model id for a given provider_id and model_type, or None."""
    for model in ogx_client.models.list().data:
        metadata = getattr(model, "custom_metadata", None) or {}
        if metadata.get("provider_id") == provider_id and metadata.get("model_type") == model_type:
            return model.id
    return None


def _resolve_non_gemini_llm(ogx_client: OgxClient) -> tuple[str, str] | None:
    """Return ``(model_id, provider_id)`` of the first non-Gemini LLM, or None.

    Selecting a model whose ``provider_id`` is explicitly not the Gemini provider
    guarantees TC-REG-002 exercises a *different* provider than remote::gemini,
    rather than accidentally routing back through Gemini.
    """
    for model in ogx_client.models.list().data:
        metadata = getattr(model, "custom_metadata", None) or {}
        provider_id = metadata.get("provider_id")
        if metadata.get("model_type") == "llm" and provider_id and provider_id != GEMINI_PROVIDER_ID:
            return model.id, provider_id
    return None


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ogx_server",
    [
        pytest.param(
            {"name": "test-gemini-regression", "randomize_name": True},
            {"enable_gemini": True},
            id="gemini",
        ),
    ],
    indirect=True,
)
class TestGeminiRegression:
    """Existing providers remain functional after adding remote::gemini."""

    @pytest.mark.tier1
    def test_remote_openai_unaffected(self, ogx_client: OgxClient) -> None:
        """Verify remote::openai still works alongside remote::gemini (TC-REG-001).

        Given: a distribution with both remote::openai and remote::gemini available.
        When: chat and embedding requests are sent through remote::openai models.
        Then: both succeed with valid responses.
        """
        # Guard the coexistence premise: if remote::gemini is not active this
        # test would otherwise pass on a plain OpenAI-only distribution and give
        # false confidence that OpenAI is "unaffected by Gemini".
        provider_types = list_provider_types(ogx_client=ogx_client)
        assert GEMINI_PROVIDER_TYPE in provider_types, (
            f"remote::gemini must be active for the TC-REG-001 coexistence scenario; got {provider_types!r}"
        )

        openai_llm = _resolve_model_id(ogx_client=ogx_client, provider_id="openai", model_type="llm")
        if not openai_llm:
            pytest.fail(reason="No remote::openai LLM model configured; cannot assert TC-REG-001")

        chat = ogx_client.chat.completions.create(
            model=openai_llm,
            messages=[{"role": "user", "content": "Just respond ACK."}],
            temperature=0,
        )
        assert chat.choices and chat.choices[0].message.content, "remote::openai chat completion failed"

        openai_embedding = _resolve_model_id(ogx_client=ogx_client, provider_id="openai", model_type="embedding")
        if openai_embedding:
            embeddings = ogx_client.embeddings.create(model=openai_embedding, input="regression check")
            assert embeddings.data and isinstance(embeddings.data[0].embedding, list), (
                "remote::openai embeddings failed"
            )
        else:
            LOGGER.info("No remote::openai embedding model configured; skipping the embeddings half of TC-REG-001")

    @pytest.mark.tier2
    def test_other_providers_unaffected(self, ogx_client: OgxClient) -> None:
        """Verify non-Gemini providers remain listed and functional (TC-REG-002).

        Given: a distribution with remote::gemini plus the default (vLLM) providers.
        When: the provider list is queried and inference is run through a model that
            is verified to belong to a non-Gemini provider.
        Then: remote::gemini is listed alongside the pre-existing providers, and the
            non-Gemini inference request returns a valid response.
        """
        provider_types = list_provider_types(ogx_client=ogx_client)
        assert GEMINI_PROVIDER_TYPE in provider_types, "remote::gemini should be listed"
        non_gemini = [ptype for ptype in provider_types if ptype != GEMINI_PROVIDER_TYPE]
        assert non_gemini, "No providers other than remote::gemini are listed"

        resolved = _resolve_non_gemini_llm(ogx_client=ogx_client)
        if not resolved:
            pytest.fail(reason="No non-Gemini LLM model is registered; cannot exercise TC-REG-002 inference")
        model_id, provider_id = resolved
        LOGGER.info(f"Running TC-REG-002 inference through non-Gemini model {model_id!r} (provider {provider_id!r})")

        response = ogx_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Just respond ACK."}],
            temperature=0,
        )
        assert response.choices and response.choices[0].message.content, (
            f"Inference through non-Gemini provider {provider_id!r} failed"
        )
