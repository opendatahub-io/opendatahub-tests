"""MaaS token passthrough via the genai-bff-proxy remote::passthrough provider.

Covers the per-request credential flow the Gen AI BFF relies on: OGX must
forward the user's MaaS token (delivered via the ``X-OGX-Provider-Data``
header) to the upstream as ``Authorization: Bearer ...`` — without OGX
understanding or storing it — together with the mapped context headers, and
must never forward a user credential that was not supplied.

Assertions are made against the mock upstream's recorded headers
(``GET /v1/debug/last-headers``), which is the only place the forwarded
credentials are observable.
"""

import hmac
import json

import httpx
import pytest
from ogx_client import APIStatusError, OgxClient

from tests.ogx.constants import OGX_CLIENT_VERIFY_SSL
from tests.ogx.mockupstream.constants import MOCK_MAAS_QUALIFIED_MODEL_ID
from tests.ogx.mockupstream.deployments import MockUpstream

USER_TOKEN = "test-user-token-maas-passthrough"

MAAS_PASSTHROUGH_NAMESPACE = {"name": "test-ogx-maas-passthrough", "randomize_name": True}

PROVIDER_DATA_WITH_TOKEN = {
    "passthrough_api_key": USER_TOKEN,
    "maas_subscription": "sub-42",
    "inference_model_source_type": "maas",
    "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
}


def _chat_completion(ogx_client: OgxClient, content: str, provider_data: dict | None = None) -> str:
    """Send a chat completion through OGX, optionally with X-OGX-Provider-Data."""
    headers = {}
    if provider_data is not None:
        headers["X-OGX-Provider-Data"] = json.dumps(provider_data)
    response = ogx_client.chat.completions.create(
        model=MOCK_MAAS_QUALIFIED_MODEL_ID,
        messages=[{"role": "user", "content": content}],
        extra_headers=headers or None,
    )
    return response.choices[0].message.content


def _get_last_request(mock_maas_upstream: MockUpstream) -> dict:
    """Return the last request the mock MaaS upstream received."""
    with httpx.Client(
        verify=OGX_CLIENT_VERIFY_SSL,
        timeout=60,
        headers=mock_maas_upstream.debug_headers(),
    ) as client:
        response = client.get(f"{mock_maas_upstream.route_url}/v1/debug/last-headers")
        response.raise_for_status()
        return response.json()


def _get_chat_completion_requests(mock_maas_upstream: MockUpstream) -> list[dict]:
    """Return the chat-completion requests the mock MaaS upstream received."""
    with httpx.Client(
        verify=OGX_CLIENT_VERIFY_SSL,
        timeout=60,
        headers=mock_maas_upstream.debug_headers(),
    ) as client:
        response = client.get(f"{mock_maas_upstream.route_url}/v1/debug/requests")
        response.raise_for_status()
        return [request for request in response.json()["data"] if request["path"] == "/v1/chat/completions"]


@pytest.mark.ogx
@pytest.mark.parametrize(
    "unprivileged_model_namespace",
    [pytest.param(MAAS_PASSTHROUGH_NAMESPACE)],
    indirect=True,
)
class TestMaasTokenPassthrough:
    """Per-request MaaS credential flow through remote::passthrough."""

    @pytest.mark.tier1
    def test_user_token_forwarded_as_authorization(
        self, ogx_client: OgxClient, mock_maas_upstream: MockUpstream
    ) -> None:
        """Verify the per-request token reaches the upstream as a Bearer credential.

        Given: an OGX server whose only inference provider is the
            genai-bff-proxy remote::passthrough provider with an empty api_key.
        When: a chat completion is sent with the user token in
            X-OGX-Provider-Data (passthrough_api_key).
        Then: the upstream receives the token verbatim as
            ``Authorization: Bearer <token>`` and the completion succeeds.
        """
        content = _chat_completion(
            ogx_client=ogx_client,
            content="hello from token passthrough test",
            provider_data={"passthrough_api_key": USER_TOKEN},
        )
        assert content == "mock-maas echo: hello from token passthrough test", f"unexpected completion: {content}"

        last_request = _get_last_request(mock_maas_upstream=mock_maas_upstream)
        authorization = last_request["headers"].get("authorization") or ""
        assert authorization, "No Authorization header forwarded to upstream"
        # The failure message intentionally omits the received value: it could
        # hold a credential that must not end up in the test report.
        assert hmac.compare_digest(authorization, f"Bearer {USER_TOKEN}"), "Authorization mismatch"

    @pytest.mark.tier1
    def test_forward_headers_reach_upstream(self, ogx_client: OgxClient, mock_maas_upstream: MockUpstream) -> None:
        """Verify X-OGX-Provider-Data keys are mapped to upstream headers.

        Given: the provider's forward_headers mapping (maas_subscription,
            inference_model_source_type, traceparent, ...).
        When: a chat completion carries those keys in X-OGX-Provider-Data.
        Then: the upstream receives them as X-MaaS-Subscription,
            X-Inference-Model-Source-Type and traceparent, alongside the Bearer
            token.
        """
        _chat_completion(
            ogx_client=ogx_client,
            content="with forwarded context",
            provider_data=PROVIDER_DATA_WITH_TOKEN,
        )

        headers = _get_last_request(mock_maas_upstream=mock_maas_upstream)["headers"]
        assert headers.get("x-maas-subscription") == "sub-42", (
            f"X-MaaS-Subscription: {headers.get('x-maas-subscription')!r}"
        )
        assert headers.get("x-inference-model-source-type") == "maas", (
            f"X-Inference-Model-Source-Type: {headers.get('x-inference-model-source-type')!r}"
        )
        assert headers.get("traceparent") == PROVIDER_DATA_WITH_TOKEN["traceparent"], "traceparent not forwarded"
        authorization = headers.get("authorization") or ""
        assert authorization, "No Authorization header forwarded to upstream"
        assert hmac.compare_digest(authorization, f"Bearer {USER_TOKEN}")

    @pytest.mark.tier3
    def test_no_token_forwarded_without_provider_data(
        self, ogx_client: OgxClient, mock_maas_upstream: MockUpstream
    ) -> None:
        """Verify no user credential leaks upstream when none was supplied.

        Given: the passthrough provider has an empty api_key.
        When: a chat completion is sent without X-OGX-Provider-Data.
        Then: either OGX refuses the request (it currently answers 500 when
            no credential can be resolved for the passthrough provider —
            opaque, but not a leak) or, if the upstream accepts it, the
            recorded request carries no Authorization header — in no case
            is a credential invented. The property is asserted against the
            mock's recordings, so it holds regardless of the refusal's
            status code.
        """
        requests_before = _get_chat_completion_requests(mock_maas_upstream=mock_maas_upstream)
        requests_before_count = len(requests_before)

        try:
            _chat_completion(ogx_client=ogx_client, content="anonymous request")
        except APIStatusError:
            # OGX refuses the request when no credential can be resolved for
            # the passthrough provider. Acceptable at any status: the leak
            # property is verified against the recordings below. Transport-
            # level failures are not caught and fail the test.
            pass

        requests_after = _get_chat_completion_requests(mock_maas_upstream=mock_maas_upstream)
        for request in requests_after[requests_before_count:]:
            # The failure message intentionally omits the received value: it
            # could hold a credential that must not end up in the test report.
            assert "authorization" not in request["headers"], "Authorization forwarded without provider data"
