from __future__ import annotations

import json

from tests.model_serving.model_server.maas_billing.utils import (
    mint_token,
    b64url_decode,
)


def test_minted_token_generated(
    http,
    base_url: str,
    maas_user_token_for_api_calls: str,
) -> None:
    """Smoke: a MaaS token can be minted (no JWT shape checks)."""
    resp, body = mint_token(
        base_url=base_url,
        oc_user_token=maas_user_token_for_api_calls,
        minutes=10,
        http=http,
    )
    assert resp.status_code in (200, 201), f"mint failed: {resp.status_code} {resp.text[:200]}"
    tok = body.get("token", "")
    print(f"[debug] MaaS token (truncated): {tok[:12]}...{tok[-12:]}")
    assert isinstance(tok, str) and len(tok) > 10, f"no usable token in response: {body}"


def test_minted_token_is_jwt(http, base_url: str, oc_user_token: str) -> None:
    resp, body = mint_token(
        base_url=base_url,
        oc_user_token=oc_user_token,
        minutes=10,
        http=http,
    )
    assert resp.status_code in (200, 201), f"mint failed: {resp.status_code} {resp.text[:200]}"

    token = body.get("token", "")
    parts = token.split(".")
    assert len(parts) == 3, "not a JWT (expected header.payload.signature)"

    header_json = b64url_decode(parts[0]).decode("utf-8")
    header = json.loads(header_json)
    assert isinstance(header, dict), "JWT header not a JSON object"
