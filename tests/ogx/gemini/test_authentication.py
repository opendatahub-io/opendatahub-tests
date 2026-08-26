"""Per-request API key override tests for the remote::gemini provider.

Covers test cases TC-AUTH-001 and TC-AUTH-002 from the remote_gemini_provider
test plan (RHAISTRAT-1245).
"""

import pytest
import structlog
from ogx_client import APIStatusError, OgxClient

from tests.ogx.constants import GEMINI_API_KEY_SECONDARY
from tests.ogx.gemini.utils import provider_data_headers

LOGGER = structlog.get_logger(name=__name__)

# These tests require live Gemini API access and must not run on disconnected clusters.
pytestmark = [pytest.mark.skip_on_disconnected]


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ogx_server",
    [
        pytest.param(
            {"name": "test-gemini-auth", "randomize_name": True},
            {"enable_gemini": True},
            id="gemini",
        ),
    ],
    indirect=True,
)
class TestGeminiPerRequestAuth:
    """Per-request Gemini API key override via the x-ogx-provider-data header."""

    @pytest.mark.tier2
    def test_per_request_api_key_override(
        self,
        ogx_client: OgxClient,
        gemini_model_id: str,
    ) -> None:
        """Verify a per-request key override authenticates the request (TC-AUTH-001).

        Given: an active remote::gemini provider with a config-level key and a
            secondary valid key.
        When: a request is sent without the header, then with x-ogx-provider-data
            carrying the secondary key.
        Then: both requests succeed, showing the per-request override is honored
            without affecting the config-level key.
        """
        if not GEMINI_API_KEY_SECONDARY:
            pytest.fail(reason="OGX_CORE_GEMINI_API_KEY_SECONDARY not set; cannot test per-request key override")

        # Baseline: config-level key.
        baseline = ogx_client.chat.completions.create(
            model=gemini_model_id,
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert baseline.choices and baseline.choices[0].message.content, "Baseline request did not succeed"

        # Per-request override with the secondary key.
        overridden = ogx_client.chat.completions.create(
            model=gemini_model_id,
            messages=[{"role": "user", "content": "Hello"}],
            extra_headers=provider_data_headers(gemini_api_key=GEMINI_API_KEY_SECONDARY),
        )
        assert overridden.choices and overridden.choices[0].message.content, (
            "Request with per-request key override did not succeed"
        )

    @pytest.mark.tier3
    def test_invalid_per_request_api_key_errors(
        self,
        ogx_client: OgxClient,
        gemini_model_id: str,
    ) -> None:
        """Verify an invalid per-request key errors without corrupting config key (TC-AUTH-002).

        Given: an active remote::gemini provider with a valid config-level key.
        When: a request is sent with an invalid key in x-ogx-provider-data, then a
            follow-up request is sent without the header.
        Then: the invalid-key request raises an API error, and the follow-up using
            the config-level key succeeds.
        """
        with pytest.raises(APIStatusError) as exc_info:
            ogx_client.chat.completions.create(
                model=gemini_model_id,
                messages=[{"role": "user", "content": "Hello"}],
                extra_headers=provider_data_headers(gemini_api_key="invalid-key-12345"),
            )

        # The failure must be a client-side rejection of the bad key (4xx), not a
        # transient 5xx/connection error that pytest.raises(APIError) would also accept.
        status_code = exc_info.value.status_code
        assert 400 <= status_code < 500, f"Expected a 4xx client error for an invalid key, got {status_code}"
        # The error payload should identify an authentication / invalid-argument problem
        # rather than an unrelated failure. Match tolerantly to avoid coupling to exact
        # provider wording, but still verify the error is about the key.
        error_text = str(exc_info.value.body or exc_info.value).lower()
        assert any(marker in error_text for marker in ("api_key", "api key", "invalid", "argument", "auth")), (
            f"Invalid-key error did not describe an authentication/argument problem: {error_text!r}"
        )

        follow_up = ogx_client.chat.completions.create(
            model=gemini_model_id,
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert follow_up.choices and follow_up.choices[0].message.content, (
            "Follow-up request with the config-level key did not succeed after an invalid override"
        )
