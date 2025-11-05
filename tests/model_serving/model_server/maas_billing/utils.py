from typing import Dict

import base64
import requests
from json import JSONDecodeError
from ocp_resources.ingress_config_openshift_io import Ingress as IngressConfig
from requests import Response
from urllib.parse import urlparse
from ocp_resources.llm_inference_service import LLMInferenceService
from utilities.llmd_utils import get_llm_inference_url


def host_from_ingress_domain(client) -> str:
    """Return 'maas.<ingress-domain>'"""
    ingress_config = IngressConfig(name="cluster", client=client, ensure_exists=True)
    domain = ingress_config.instance.spec.get("domain")
    assert domain, "Ingress 'cluster' missing spec.domain (ingresses.config.openshift.io)"
    return f"maas.{domain}"


def detect_scheme_via_llmisvc(client, namespace: str = "llm") -> str:
    """
    Using LLMInferenceService's URLto infer the scheme.
    """
    for llm in LLMInferenceService.get(dyn_client=client, namespace=namespace):
        conditions = llm.instance.status.get("conditions", [])
        if any(c.get("type") == "Ready" and c.get("status") == "True" for c in conditions):
            url = get_llm_inference_url(llm=llm)
            scheme = (urlparse(url).scheme or "").lower()
            if scheme in ("http", "https"):
                return scheme
    return "http"


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
