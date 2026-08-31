"""Tests for remote::gemini provider availability and activation.

Covers test cases TC-PROV-001, TC-PROV-002 and TC-PROV-003 from the
remote_gemini_provider test plan (RHAISTRAT-1245).
"""

import pytest
import structlog
from ogx_client import OgxClient

from tests.ogx.constants import GEMINI_PROVIDER_TYPE
from tests.ogx.gemini.utils import is_gemini_provider_active, list_provider_types

LOGGER = structlog.get_logger(name=__name__)

# These tests require live Gemini API access and must not run on disconnected clusters.
pytestmark = [pytest.mark.skip_on_disconnected]


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ogx_server",
    [
        pytest.param(
            {"name": "test-gemini-providers", "randomize_name": True},
            {"enable_gemini": True},
            id="gemini",
        ),
    ],
    indirect=True,
)
class TestGeminiProviders:
    """Provider availability and conditional activation for remote::gemini."""

    @pytest.mark.tier1
    def test_gemini_provider_listed(self, ogx_client: OgxClient) -> None:
        """Verify remote::gemini is listed in /v1/providers (TC-PROV-001).

        Given: an OgxServer deployed with GEMINI_API_KEY injected.
        When: the provider list is retrieved from /v1/providers.
        Then: a provider entry with provider_type "remote::gemini" is present.
        """
        provider_types = list_provider_types(ogx_client=ogx_client)
        LOGGER.info(f"Providers reported by the distribution: {provider_types}")
        assert GEMINI_PROVIDER_TYPE in provider_types, (
            f"Expected {GEMINI_PROVIDER_TYPE!r} in provider list, got {provider_types!r}"
        )

    @pytest.mark.tier1
    def test_gemini_provider_no_network_headers_workaround(
        self,
        ogx_client: OgxClient,
        gemini_model_id: str,
    ) -> None:
        """Verify remote::gemini works without a network.headers workaround (TC-PROV-002).

        Given: a distribution configured with provider_type remote::gemini.
        When: the provider config is inspected and a chat completion is sent.
        Then: no network.headers override is present and the completion succeeds.
        """
        gemini_provider = next(
            provider for provider in ogx_client.providers.list() if provider.provider_type == GEMINI_PROVIDER_TYPE
        )
        provider_config = getattr(gemini_provider, "config", None) or {}
        network_config = provider_config.get("network", {}) if isinstance(provider_config, dict) else {}
        assert "headers" not in network_config, (
            f"Unexpected network.headers workaround in remote::gemini config: {network_config!r}"
        )

        response = ogx_client.chat.completions.create(
            model=gemini_model_id,
            messages=[{"role": "user", "content": "Just respond ACK."}],
            temperature=0,
        )
        assert response.choices, "No choices returned from Gemini chat completion"
        assert response.choices[0].message.content, "Gemini chat completion returned empty content"

    @pytest.mark.tier1
    def test_gemini_conditional_activation_with_key(self, ogx_client: OgxClient) -> None:
        """Verify remote::gemini activates when GEMINI_API_KEY is set (TC-PROV-003).

        Given: an OgxServer deployed with GEMINI_API_KEY set.
        When: /v1/providers is queried.
        Then: remote::gemini is active.

        Note: the negative half of TC-PROV-003 (redeploying the distribution
        *without* GEMINI_API_KEY and asserting the provider disappears) is a
        destructive re-rollout of a class-scoped server and is intentionally not
        automated here; it requires a dedicated no-key deployment. Track as a
        follow-up before marking TC-PROV-003 fully automated.
        """
        assert is_gemini_provider_active(ogx_client=ogx_client), (
            "remote::gemini should be active when GEMINI_API_KEY is injected"
        )
