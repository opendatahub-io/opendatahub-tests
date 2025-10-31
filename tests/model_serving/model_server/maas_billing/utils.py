from __future__ import annotations

from typing import Dict, Optional

import base64
import requests
from ocp_resources.gateway_gateway_networking_k8s_io import Gateway
from ocp_resources.ingress_config_openshift_io import Ingress as IngressConfig
from pyhelper_utils.shell import run_command
from requests import Response


def host_from_ingress_domain(client) -> str:
    """
    Return 'maas.<ingress-domain>' using the wrapper (no 'oc get' calls).
    """
    ing = IngressConfig(name="cluster", client=client)
    assert ing.exists, "ingresses.config.openshift.io/cluster not found"

    spec = ing.instance.spec
    domain = spec.get("domain") if isinstance(spec, dict) else getattr(spec, "domain", None)
    assert domain, "spec.domain is missing on Ingress 'cluster'"

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
    """Cluster-wide; prefer maas-default-gateway if present, else first."""
    gateways = list(Gateway.get(client=client))
    if not gateways:
        return "http"

    gw = next(
        (g for g in gateways if (g.instance.metadata.name or "") == "maas-default-gateway"),
        gateways[0],
    )
    return scheme_from_gateway(gw=gw)


def current_user_bearer_via_oc() -> str:
    """
    Return the current oc login token from `oc whoami -t`.
    """
    rc, out, err = run_command(command=["oc", "whoami", "-t"])
    assert (rc is True) or (rc == 0), f"failed to get token via 'oc whoami -t': rc={rc} err={err}"
    token = (out or "").strip()
    assert token, "empty token from 'oc whoami -t'"
    return token


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
