from typing import Dict, Generator

import base64
import requests
import tempfile
import shlex
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
from timeout_sampler import TimeoutSampler
from ocp_resources.deployment import Deployment
from pathlib import Path
from pyhelper_utils.shell import run_command
from utilities.infra import login_with_user_password

LOGGER = get_logger(name=__name__)


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


def wait_for_oauth_openshift_deployment() -> None:
    """
    Wait for the oauth-openshift deployment in openshift-authentication
    to finish rolling out after we update OAuth identityProviders.

    """
    oauth_deployment = Deployment(
        name="oauth-openshift",
        namespace="openshift-authentication",
        ensure_exists=True,
    )

    LOGGER.info("Waiting for oauth-openshift rollout to finish")

    def _get_conditions():
        conditions = getattr(oauth_deployment.instance.status, "conditions", []) or []
        details = [(condition.type, condition.status, getattr(condition, "reason", "")) for condition in conditions]
        LOGGER.info(f"oauth-openshift conditions: {details}")
        return conditions

    # Quick check first – if it's already Available=True
    initial_conditions = _get_conditions()
    if any(condition.type == "Available" and condition.status == "True" for condition in initial_conditions):
        LOGGER.info("oauth-openshift already Available=True; not waiting further for Progressing to clear")
        return

    # Otherwise, wait up to 5 minutes for Available=True
    sampler = TimeoutSampler(
        wait_timeout=300,
        sleep=5,
        func=_get_conditions,
    )

    for conditions in sampler:
        if any(condition.type == "Available" and condition.status == "True" for condition in conditions):
            LOGGER.info("oauth-openshift became Available=True")
            return


def make_bcrypt_htpasswd_file_with_users(users: list[tuple[str, str]]) -> Path:
    """
    Create a single htpasswd file (-B bcrypt) containing multiple users.
    `users` is a list of (username, password) tuples.
    """
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        htpasswd_path = Path(temp_file.name).resolve()

    # First user: create (-c)
    first_user, first_pass = users[0]
    run_command(
        command=shlex.split(f"htpasswd -c -B -b {htpasswd_path} {first_user} {first_pass}"),
        check=True,
    )

    # Remaining users: append (no -c)
    for username, password_value in users[1:]:
        run_command(
            command=shlex.split(f"htpasswd -B -b {htpasswd_path} {username} {password_value}"),
            check=True,
        )
    return htpasswd_path


def login_with_retry(
    api: str,
    user: str,
    password: str | None = None,
    wait_timeout: int = 60,
    sleep: float = 2.0,
) -> None:
    """
    Login helper that retries a few times in case the cluster is not ready.
    This avoids test failures caused by temporary login errors.
    """
    last_exc: Exception | None = None

    def _attempt_login() -> bool:
        nonlocal last_exc
        try:
            login_with_user_password(api_address=api, user=user, password=password)
            return True
        except Exception as login_error:  # noqa: BLE001
            last_exc = login_error
            error_text = str(login_error) or "<no error message>"
            LOGGER.warning(f"MaaS RBAC: login failed for {user} ({error_text}); will retry")
            return False

    sampler = TimeoutSampler(
        wait_timeout=wait_timeout,
        sleep=sleep,
        func=_attempt_login,
    )

    for ok in sampler:
        if ok:
            LOGGER.info(f"MaaS RBAC: login succeeded for {user}")
            return

    # If we exit the loop without success, timeout was hit
    raise last_exc if last_exc else RuntimeError(f"Login failed for user {user}")
