from typing import Dict, Generator

import base64
import requests
from json import JSONDecodeError
from ocp_resources.ingress_config_openshift_io import Ingress as IngressConfig
from requests import Response
from urllib.parse import urlparse
from ocp_resources.llm_inference_service import LLMInferenceService
from utilities.llmd_utils import get_llm_inference_url

from contextlib import contextmanager
from kubernetes.dynamic import DynamicClient
from ocp_resources.group import Group
from simple_logger.logger import get_logger
from utilities.plugins.constant import RestHeader, OpenAIEnpoints

LOGGER = get_logger(name=__name__)
MODELS_INFO = OpenAIEnpoints.MODELS_INFO


def host_from_ingress_domain(client) -> str:
    """Return 'maas.<ingress-domain>'"""
    ingress_config = IngressConfig(name="cluster", client=client, ensure_exists=True)
    domain = ingress_config.instance.spec.get("domain")
    assert domain, "Ingress 'cluster' missing spec.domain (ingresses.config.openshift.io)"
    return f"maas.{domain}"


def _first_ready_llmisvc(
    client,
    namespace: str = "llm",
    label_selector: str | None = None,
):
    """
    Return the first Ready LLMInferenceService in the given namespace,
    or None if none are Ready.
    """
    for service in LLMInferenceService.get(
        dyn_client=client,
        namespace=namespace,
        label_selector=label_selector,
    ):
        status = getattr(service.instance, "status", {}) or {}
        conditions = status.get("conditions", [])
        is_ready = any(
            condition.get("type") == "Ready" and condition.get("status") == "True" for condition in conditions
        )
        if is_ready:
            return service

    return None


def detect_scheme_via_llmisvc(client, namespace: str = "llm") -> str:
    """
    Using LLMInferenceService's URL to infer the scheme.
    """
    service = _first_ready_llmisvc(client=client, namespace=namespace)
    if not service:
        return "http"

    url = get_llm_inference_url(llm_service=service)
    scheme = (urlparse(url).scheme or "").lower()
    if scheme in ("http", "https"):
        return scheme

    return "http"


def maas_auth_headers(token: str) -> Dict[str, str]:
    """Authorization header only (used for /v1/tokens with OCP user token)."""
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


def llmis_name(client, namespace: str = "llm", label_selector: str | None = None) -> str:
    """
    Return the name of the first Ready LLMInferenceService.
    """

    service = _first_ready_llmisvc(
        client=client,
        namespace=namespace,
        label_selector=label_selector,
    )
    if not service:
        raise RuntimeError("No Ready LLMInferenceService found")

    return service.name


@contextmanager
def create_maas_group(
    admin_client: DynamicClient,
    group_name: str,
    users: list[str] | None = None,
) -> Generator[Group, None, None]:
    """
    Create an OpenShift Group with optional users and delete it on exit.
    """
    with Group(
        client=admin_client,
        name=group_name,
        users=users or [],
        wait_for_resource=True,
    ) as group:
        LOGGER.info(f"MaaS RBAC: created group {group_name} with users {users or []}")
        yield group


def build_maas_headers(token: str) -> dict:
    """Return common MaaS headers for a given token."""
    return {
        "Authorization": f"Bearer {token}",
        **RestHeader.HEADERS,
    }


def get_maas_models_response(
    session: requests.Session,
    base_url: str,
    headers: dict,
) -> requests.Response:
    """
    Issue GET /v1/models and return the raw Response.

    Also validates the status code before returning.
    """
    models_url = f"{base_url}{MODELS_INFO}"
    resp = session.get(url=models_url, headers=headers, timeout=60)

    LOGGER.info(f"MaaS: /v1/models -> {resp.status_code} (url={models_url})")

    assert resp.status_code == 200, f"/v1/models failed: {resp.status_code} {resp.text[:200]} (url={models_url})"

    return resp
