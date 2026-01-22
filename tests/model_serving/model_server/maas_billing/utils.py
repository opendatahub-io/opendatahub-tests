from typing import Any, Dict, Generator, List, Tuple
import base64
import requests
from json import JSONDecodeError
from urllib.parse import urlparse
from contextlib import contextmanager

from kubernetes.dynamic import DynamicClient
from ocp_resources.group import Group
from ocp_resources.ingress_config_openshift_io import Ingress as IngressConfig
from ocp_resources.llm_inference_service import LLMInferenceService
from requests import Response
from simple_logger.logger import get_logger
from utilities.llmd_utils import get_llm_inference_url
from utilities.plugins.constant import RestHeader, OpenAIEnpoints
from ocp_resources.resource import ResourceEditor
from utilities.resources.rate_limit_policy import RateLimitPolicy
from utilities.resources.token_rate_limit_policy import TokenRateLimitPolicy

# from ocp_resources.gateway_gateway_networking_k8s_io import Gateway
from ocp_resources.endpoints import Endpoints
from ocp_resources.config_map import ConfigMap
from ocp_resources.namespace import Namespace
from utilities.constants import (
    MAAS_GATEWAY_NAME,
    MAAS_GATEWAY_NAMESPACE,
)


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
        client=client,
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


@contextmanager
def patch_llmisvc_with_maas_router(
    llm_service: LLMInferenceService,
) -> Generator[None, None, None]:
    router_spec = {
        "gateway": {"refs": [{"name": MAAS_GATEWAY_NAME, "namespace": MAAS_GATEWAY_NAMESPACE}]},
        "route": {},
    }

    patch_body = {
        "metadata": {
            "annotations": {
                "alpha.maas.opendatahub.io/tiers": "[]",
                "security.opendatahub.io/enable-auth": "true",
            }
        },
        "spec": {"router": router_spec},
    }

    with ResourceEditor(patches={llm_service: patch_body}):
        yield


def verify_chat_completions(
    request_session_http: requests.Session,
    model_url: str,
    headers: dict,
    models_list: list,
    *,
    prompt_text: str = "Hello from MaaS chat e2e test",
    max_tokens: int = 50,
    request_timeout_seconds: int = 60,
    log_prefix: str = "MaaS",
    expected_status_codes: tuple[int, ...] = (200,),
) -> Response:
    """
    Common helper to verify /v1/chat/completions responds to a simple prompt.

    - For the usual happy-path tests, leave expected_status_codes=(200,)
      and this behaves exactly as before: assert HTTP 200 and validate the
      basic response shape (choices, content, etc.).

    - For special tests (e.g. rate limiting) you can pass a tuple like
      expected_status_codes=(200, 429) and then inspect response.status_code
      in the test. Only HTTP 200 responses will have their body validated.
    """

    assert models_list, "No models returned from /v1/models"
    first_model = models_list[0]

    model_id = first_model.get("id", "")
    assert model_id, "First model from /v1/models has no 'id' field"

    payload_data = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": max_tokens,
    }

    LOGGER.info(f"{log_prefix}: POST {model_url} with payload keys={list(payload_data.keys())}")

    response = request_session_http.post(
        url=model_url,
        headers=headers,
        json=payload_data,
        timeout=request_timeout_seconds,
    )

    LOGGER.info(f"{log_prefix}: POST {model_url} -> HTTP {response.status_code}")

    assert response.status_code in expected_status_codes, (
        f"/v1/chat/completions failed: HTTP {response.status_code} "
        f"response={response.text[:200]} (url={model_url}), "
        f"expected one of {expected_status_codes}"
    )

    if response.status_code == 200:
        response_body = response.json()
        completions_choices = response_body.get("choices", [])
        assert isinstance(completions_choices, list) and completions_choices, (
            "'choices' field missing or empty in /v1/chat/completions response"
        )

        first_choice = completions_choices[0]
        message_section = first_choice.get("message", {}) or {}
        content_text = message_section.get("content") or first_choice.get("text", "")

        assert isinstance(content_text, str) and content_text.strip(), (
            "First choice in /v1/chat/completions response has no text content"
        )

    return response


def assert_mixed_200_and_429(
    *,
    actor_label: str,
    status_codes_list: List[int],
    context: str,
    require_429: bool = True,
) -> None:
    assert status_codes_list, f"{actor_label}: no responses in {context}"

    allowed_status_codes = {200, 429}
    unexpected_status_codes = [code for code in status_codes_list if code not in allowed_status_codes]

    assert not unexpected_status_codes, (
        f"{actor_label}: unexpected HTTP status codes in {context}: {unexpected_status_codes}. "
        f"Full sequence={status_codes_list}. "
        "Likely wrong endpoint/route (404) or server/gateway issue (5xx)."
    )

    assert status_codes_list[0] == 200, (
        f"{actor_label}: expected first status 200 in {context}, got {status_codes_list[0]} "
        f"(status_codes={status_codes_list})"
    )

    if require_429:
        assert 429 in status_codes_list, f"{actor_label}: expected 429 in {context}, but saw {status_codes_list}"

        first_429_idx = status_codes_list.index(429)
        start = first_429_idx + 1
        tail_after_429 = status_codes_list[start:]

        assert 200 not in tail_after_429, (
            f"{actor_label}: saw 200 after first 429 in {context}. "
            f"First 429 at index {first_429_idx}. Full sequence={status_codes_list}"
        )


def maas_token_ratelimitpolicy_spec() -> Dict[str, Any]:
    """
    Deterministic TokenRateLimitPolicy limits for MaaS tests.

    This returns only the 'limits' mapping, suitable for assigning to
    spec['limits'] on the existing TokenRateLimitPolicy.
    """
    return {
        "enterprise-user-tokens": {
            "counters": [{"expression": "auth.identity.userid"}],
            "rates": [{"limit": 240, "window": "1m"}],
            "when": [{"predicate": 'auth.identity.tier == "enterprise"'}],
        },
        "free-user-tokens": {
            "counters": [{"expression": "auth.identity.userid"}],
            "rates": [{"limit": 60, "window": "1m"}],
            "when": [{"predicate": 'auth.identity.tier == "free"'}],
        },
        "premium-user-tokens": {
            "counters": [{"expression": "auth.identity.userid"}],
            "rates": [{"limit": 120, "window": "1m"}],
            "when": [{"predicate": 'auth.identity.tier == "premium"'}],
        },
    }


def maas_ratelimitpolicy_spec() -> Dict[str, Any]:
    """
    Deterministic RateLimitPolicy limits for MaaS tests.

    This returns only the 'limits' mapping, suitable for assigning to
    spec['limits'] on the existing RateLimitPolicy.
    """
    return {
        "enterprise": {
            "counters": [{"expression": "auth.identity.userid"}],
            "rates": [{"limit": 50, "window": "2m"}],
            "when": [{"predicate": 'auth.identity.tier == "enterprise"'}],
        },
        "free": {
            "counters": [{"expression": "auth.identity.userid"}],
            "rates": [{"limit": 5, "window": "1m"}],
            "when": [{"predicate": 'auth.identity.tier == "free"'}],
        },
        "premium": {
            "counters": [{"expression": "auth.identity.userid"}],
            "rates": [{"limit": 8, "window": "1m"}],
            "when": [{"predicate": 'auth.identity.tier == "premium"'}],
        },
    }


@contextmanager
def maas_gateway_rate_limits_patched(
    *,
    admin_client: DynamicClient,
    namespace: str,
    token_policy_name: str,
    request_policy_name: str,
) -> Generator[None, None, None]:
    """
    Temporarily patch ONLY `spec.limits` of the Kuadrant TokenRateLimitPolicy and
    RateLimitPolicy for MaaS tests, and restore the original state afterwards.
    """
    token_policy = TokenRateLimitPolicy(
        client=admin_client,
        name=token_policy_name,
        namespace=namespace,
        ensure_exists=True,
    )
    request_policy = RateLimitPolicy(
        client=admin_client,
        name=request_policy_name,
        namespace=namespace,
        ensure_exists=True,
    )

    LOGGER.info(f"MaaS Kuadrant: using policies {namespace}/{token_policy_name} and {namespace}/{request_policy_name}")

    LOGGER.info(f"Patching TokenRateLimitPolicy in namespace '{namespace}' via ResourceEditor")
    with ResourceEditor(patches={token_policy: {"spec": {"limits": maas_token_ratelimitpolicy_spec()}}}):
        LOGGER.info("Waiting for TokenRateLimitPolicy condition Enforced=True")
        token_policy.wait_for_condition(condition="Enforced", status="True", timeout=60)

        LOGGER.info(f"Patching RateLimitPolicy in namespace '{namespace}' via ResourceEditor")
        with ResourceEditor(patches={request_policy: {"spec": {"limits": maas_ratelimitpolicy_spec()}}}):
            LOGGER.info("Waiting for RateLimitPolicy condition Enforced=True")
            request_policy.wait_for_condition(condition="Enforced", status="True", timeout=60)
            yield

    LOGGER.info("Restored original Kuadrant policies")


def get_total_tokens(resp: Response, *, fail_if_missing: bool = False) -> int | None:
    """Extract total token usage from a MaaS response.

    The helper first checks for the `x-odhu-usage-total-tokens` response header.
    If it is missing or not parseable as an integer, the JSON body is used.

    Args:
        resp: HTTP response returned by a MaaS inference endpoint.
        fail_if_missing: If True, raise AssertionError when token usage cannot be extracted.

    Returns:
        Total token count if available, otherwise None.

    Raises:
        AssertionError: If fail_if_missing=True and token usage cannot be extracted.
    """
    header_val = resp.headers.get("x-odhu-usage-total-tokens")
    if header_val is not None:
        try:
            return int(header_val)
        except (TypeError, ValueError):
            if fail_if_missing:
                raise AssertionError(
                    f"Token usage header is not parseable as int; headers={dict(resp.headers)} body={resp.text[:500]}"
                ) from None
            return None

    try:
        body: Any = resp.json()
    except ValueError:
        if fail_if_missing:
            raise AssertionError(
                f"Token usage not found: response body is not JSON; headers={dict(resp.headers)} body={resp.text[:500]}"
            ) from None
        return None

    if isinstance(body, dict):
        usage = body.get("usage")
        if isinstance(usage, dict):
            total = usage.get("total_tokens")
            if isinstance(total, int):
                return total

    if fail_if_missing:
        raise AssertionError(
            f"Token usage not found in header or JSON body; headers={dict(resp.headers)} body={resp.text[:500]}"
        )
    return None


def maas_gateway_listeners(hostname: str) -> List[Dict[str, Any]]:
    return [
        {
            "name": "http",
            "hostname": hostname,
            "port": 80,
            "protocol": "HTTP",
            "allowedRoutes": {"namespaces": {"from": "All"}},
        },
        {
            "name": "https",
            "hostname": hostname,
            "port": 443,
            "protocol": "HTTPS",
            "allowedRoutes": {"namespaces": {"from": "All"}},
            "tls": {
                "mode": "Terminate",
                "certificateRefs": [{"group": "", "kind": "Secret", "name": "data-science-gateway-service-tls"}],
            },
        },
    ]


def detect_maas_control_plane_namespace(
    admin_client: DynamicClient,
    candidate_namespaces: list[str],
) -> str:
    for ns in candidate_namespaces:
        if not Namespace(client=admin_client, name=ns, ensure_exists=False).exists:
            continue

        cm = ConfigMap(
            client=admin_client,
            name="tier-to-group-mapping",
            namespace=ns,
            ensure_exists=False,
        )
        if cm.exists:
            LOGGER.info(f"MaaS control-plane namespace detected via ConfigMap in namespace: {ns}")
            return ns

    raise RuntimeError(
        "Required ConfigMap 'tier-to-group-mapping' was not found in any expected namespace. "
        f"Checked namespaces: {candidate_namespaces}"
    )


def endpoints_have_ready_addresses(
    admin_client: DynamicClient,
    namespace: str,
    name: str,
) -> bool:
    endpoints = Endpoints(
        client=admin_client,
        name=name,
        namespace=namespace,
        ensure_exists=True,
    )

    subsets = endpoints.instance.subsets
    if not subsets:
        return False

    return any(subset.addresses for subset in subsets)


def gateway_probe_reaches_maas_api(
    http_session: requests.Session,
    probe_url: str,
    request_timeout_seconds: int,
) -> Tuple[bool, int, str]:
    response = http_session.get(probe_url, timeout=request_timeout_seconds)
    status_code = response.status_code
    response_text = response.text
    ok = status_code in (200, 401, 403)
    return ok, status_code, response_text
