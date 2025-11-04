from typing import Dict

import base64
import requests
from json import JSONDecodeError
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
    HTTPS or TLS => 'https'; otherwise 'http'.
    """
    listeners = gw.instance.spec.get("listeners", [])
    for listener in listeners:
        protocol = listener["protocol"].upper()
        if protocol in ("HTTPS", "TLS"):
            return "https"
    return "http"


def choose_scheme_via_gateway(client) -> str:
    # Gateway created in namespace 'openshift-ingress'
    gw = Gateway(
        name="maas-default-gateway",
        namespace="openshift-ingress",
        client=client,
        ensure_exists=True,
    )
    return scheme_from_gateway(gw=gw)


def maas_auth_headers(token: str) -> Dict[str, str]:
    """Build Authorization header for MaaS/Billing calls."""
    return {"Authorization": f"Bearer {token}"}


def mint_token(
    base_url: str,
    oc_user_token: str,
    http_session: requests.Session,
    minutes: int = 10,
) -> tuple[Response, dict]:
    """Mint a MaaS token."""
    resp = http_session.post(
        f"{base_url}/v1/tokens",
        headers=maas_auth_headers(token=oc_user_token),
        json={"ttl": f"{minutes}m"},
        timeout=60,
    )
    try:
        body = resp.json()
    except (JSONDecodeError, ValueError):
        body = {}
    return resp, body


def b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    # keyword-arg to satisfy FCN001 rule:
    return base64.urlsafe_b64decode(s=(s + pad).encode("utf-8"))
