from __future__ import annotations

from typing import Dict, Optional

import base64
import requests
from ocp_resources.gateway_gateway_networking_k8s_io import Gateway
from ocp_resources.ingress_config_openshift_io import Ingress as IngressConfig
from requests import Response


def host_from_ingress_domain(client) -> str:
    """Return 'maas.<ingress-domain>'"""
    ingress_config = IngressConfig(name="cluster", client=client, ensure_exists=True)
    domain = ingress_config.instance.spec.get("domain")
    assert domain, "Ingress 'cluster' missing spec.domain (ingresses.config.openshift.io)"
    return f"maas.{domain}"


def scheme_from_gateway(gw: Gateway) -> str:
    """
    Decide 'http' or 'https' from Gateway listeners.
    Rule: if any listener has protocol HTTPS or tls set, return 'https'; else 'http'.
    listeners is a list[dict] per the Gateway wrapper.
    """
    listeners = gw.instance.spec.get("listeners", [])
    for listener in listeners:
        protocol = (listener.get("protocol") or "").upper()
        if protocol == "HTTPS" or listener.get("tls"):
            return "https"
    return "http"


def choose_scheme_via_gateway(client) -> str:
    """Prefer 'maas-default-gateway' if present; else first discovered; else 'http'."""
    try:
        named = Gateway(name="maas-default-gateway", client=client)
        if named.exists:
            return scheme_from_gateway(gw=named)
    except Exception:
        pass

    for gw in Gateway.get(client=client):
        return scheme_from_gateway(gw=gw)
    return "http"


def maas_auth_headers(token: str) -> Dict[str, str]:
    """Build Authorization header for MaaS/Billing calls."""
    return {"Authorization": f"Bearer {token}"}


def mint_token(
    base_url: str,
    oc_user_token: str,
    minutes: int = 10,
    http: Optional[requests.Session] = None,
) -> tuple[Response, dict]:
    """Mint a MaaS token.

    Args:
        base_url: MaaS API base, e.g. "https://maas.apps.../maas-api".
        oc_user_token: Bearer used to mint the token (usually `oc whoami -t`).
        minutes: Token TTL in minutes.

    Returns:
        (raw requests.Response, parsed_json_or_empty_dict)
    """
    assert http is not None, "HTTP session is required (pass the fixture)"
    resp = http.post(
        f"{base_url}/v1/tokens",
        headers=maas_auth_headers(token=oc_user_token),
        json={"ttl": f"{minutes}m"},
        timeout=60,
    )
    try:
        body = resp.json()
    except Exception:
        body = {}
    return resp, body


def b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    # keyword-arg to satisfy FCN001 rule:
    return base64.urlsafe_b64decode(s=(s + pad).encode("utf-8"))
