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
    Using LLMInferenceService's URL to infer the scheme.
    """
    for inference_service in LLMInferenceService.get(dyn_client=client, namespace=namespace):
        status_conditions = inference_service.instance.status.get("conditions", [])
        service_is_ready = any(
            condition_entry.get("type") == "Ready" and condition_entry.get("status") == "True"
            for condition_entry in status_conditions
        )
        if service_is_ready:
            url = get_llm_inference_url(llm_service=inference_service)
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


def b64url_decode(encoded_str: str) -> bytes:
    padding = "=" * (-len(encoded_str) % 4)
    padded_bytes = (encoded_str + padding).encode(encoding="utf-8")
    return base64.urlsafe_b64decode(s=padded_bytes)
