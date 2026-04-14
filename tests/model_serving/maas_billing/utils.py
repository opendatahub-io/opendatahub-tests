import base64
import json
from collections.abc import Generator
from contextlib import contextmanager
from json import JSONDecodeError
from typing import Any
from urllib.parse import quote, urlparse

import requests
import structlog
from kubernetes.dynamic import DynamicClient

# from ocp_resources.gateway_gateway_networking_k8s_io import Gateway
from ocp_resources.endpoints import Endpoints
from ocp_resources.group import Group
from ocp_resources.ingress_config_openshift_io import Ingress as IngressConfig
from ocp_resources.llm_inference_service import LLMInferenceService
from ocp_resources.resource import ResourceEditor
from requests import Response

from utilities.constants import (
    MAAS_GATEWAY_NAME,
    MAAS_GATEWAY_NAMESPACE,
)
from utilities.llmd_utils import get_llm_inference_url
from utilities.plugins.constant import OpenAIEnpoints, RestHeader
from utilities.resources.rate_limit_policy import RateLimitPolicy
from utilities.resources.token_rate_limit_policy import TokenRateLimitPolicy

LOGGER = structlog.get_logger(name=__name__)
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
        return "https"

    url = get_llm_inference_url(llm_service=service)
    scheme = (urlparse(url).scheme or "").lower()
    if scheme in ("http", "https"):
        return scheme

    return "https"


def maas_auth_headers(token: str) -> dict[str, str]:
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
    except JSONDecodeError, ValueError:
        body = {}
    return resp, body


def b64url_decode(encoded_str: str) -> bytes:
    padding = "=" * (-len(encoded_str) % 4)
    padded_bytes = (encoded_str + padding).encode(encoding="utf-8")
    return base64.urlsafe_b64decode(s=padded_bytes)


@contextmanager
def create_maas_group(
    admin_client: DynamicClient,
    group_name: str,
    users: list[str] | None = None,
) -> Generator[Group]:
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
) -> Generator[None]:
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
    status_codes_list: list[int],
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


def maas_token_ratelimitpolicy_spec() -> dict[str, Any]:
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


def maas_ratelimitpolicy_spec() -> dict[str, Any]:
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
) -> Generator[None]:
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
        except TypeError, ValueError:
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


def maas_gateway_listeners(hostname: str) -> list[dict[str, Any]]:
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
) -> tuple[bool, int, str]:
    response = http_session.get(probe_url, timeout=request_timeout_seconds)
    status_code = response.status_code
    response_text = response.text
    LOGGER.info(f"Received {status_code} response from {probe_url}, {response_text}")
    ok = status_code in (200, 401, 403)
    return ok, status_code, response_text


def revoke_token(
    base_url: str,
    oc_user_token: str,
    http_session: requests.Session,
) -> Response:
    """
    Revoke MaaS tokens for the user.
    """
    url = f"{base_url}/v1/tokens"
    resp = http_session.delete(
        url=url,
        headers=maas_auth_headers(token=oc_user_token),
        timeout=60,
    )

    assert resp.status_code in (200, 202, 204), f"revoke failed: {resp.status_code} {(resp.text or '')[:200]}"

    return resp


def create_api_key(
    base_url: str,
    ocp_user_token: str,
    request_session_http: requests.Session,
    api_key_name: str,
    request_timeout_seconds: int = 60,
    expires_in: str | None = None,
    raise_on_error: bool = True,
    subscription: str | None = None,
    ephemeral: bool = False,
) -> tuple[Response, dict[str, Any]]:
    """Create an API key via MaaS API and return (response, parsed_body).

    Args:
        base_url: MaaS API base URL.
        ocp_user_token: OCP token for auth against maas-api.
        request_session_http: HTTP session to use.
        api_key_name: Name of the API key to create.
        expires_in: Optional expiration duration string (e.g. "24h", "720h").
            When None, no expiresIn field is sent and the key does not expire.
        raise_on_error: When True (default), raises AssertionError for non-200/201
            responses. Set to False when testing error cases (e.g. 400 rejection).
        subscription: Optional MaaSSubscription name to bind at mint time.
        ephemeral: When True, marks the key as short-lived/programmatic.
    """
    api_keys_url = f"{base_url}/v1/api-keys"
    payload: dict[str, Any] = {"name": api_key_name}
    if expires_in is not None:
        payload["expiresIn"] = expires_in
    if subscription is not None:
        payload["subscription"] = subscription
    if ephemeral:
        payload["ephemeral"] = True

    response = request_session_http.post(
        url=api_keys_url,
        headers={"Authorization": f"Bearer {ocp_user_token}", "Content-Type": "application/json"},
        json=payload,
        timeout=request_timeout_seconds,
    )
    LOGGER.info(f"create_api_key: url={api_keys_url} status={response.status_code}")
    if response.status_code not in (200, 201):
        if raise_on_error:
            raise AssertionError(f"api-key create failed: status={response.status_code} body={response.text[:500]}")
        return response, {}

    try:
        parsed_body: dict[str, Any] = json.loads(response.text)
    except json.JSONDecodeError as error:
        raise AssertionError("API key creation returned non-JSON response") from error

    api_key = parsed_body.get("key", "")
    if not isinstance(api_key, str) or not api_key.startswith("sk-"):
        raise AssertionError("No plaintext api key returned in MaaS API response")

    return response, parsed_body


def revoke_api_key(
    request_session_http: requests.Session,
    base_url: str,
    key_id: str,
    ocp_user_token: str,
    request_timeout_seconds: int = 60,
) -> tuple[Response, dict[str, Any]]:
    """Revoke an API key via MaaS API (DELETE /v1/api-keys/{id})."""
    url = f"{base_url}/v1/api-keys/{quote(key_id, safe='')}"
    response = request_session_http.delete(
        url=url,
        headers={"Authorization": f"Bearer {ocp_user_token}"},
        timeout=request_timeout_seconds,
    )
    LOGGER.info(f"revoke_api_key: url={url} key_id={key_id} status={response.status_code}")
    try:
        parsed_body: dict[str, Any] = json.loads(response.text)
    except json.JSONDecodeError as error:
        raise AssertionError(
            f"revoke_api_key returned non-JSON response: status={response.status_code} body={response.text[:200]}"
        ) from error
    return response, parsed_body


def create_and_yield_api_key_id(
    request_session_http: requests.Session,
    base_url: str,
    ocp_user_token: str,
    key_name_prefix: str,
    expires_in: str | None = None,
) -> Generator[str]:
    """Create an API key, yield its ID, and revoke it on teardown."""
    from utilities.general import generate_random_name

    key_name = f"{key_name_prefix}-{generate_random_name()}"
    _, body = create_api_key(
        base_url=base_url,
        ocp_user_token=ocp_user_token,
        request_session_http=request_session_http,
        api_key_name=key_name,
        expires_in=expires_in,
    )
    LOGGER.info(f"create_and_yield_api_key_id: created key id={body['id']} name={key_name}")
    yield body["id"]
    LOGGER.info(f"create_and_yield_api_key_id: teardown revoking key id={body['id']}")
    revoke_resp, _ = revoke_api_key(
        request_session_http=request_session_http,
        base_url=base_url,
        key_id=body["id"],
        ocp_user_token=ocp_user_token,
    )
    if revoke_resp.status_code not in (200, 404):
        raise AssertionError(
            f"Unexpected teardown status for key id={body['id']}: {revoke_resp.status_code} {revoke_resp.text[:200]}"
        )


def assert_api_key_created_ok(
    resp: Response,
    body: dict[str, Any],
    required_fields: tuple[str, ...] = ("key",),
) -> None:
    """Assert an API key creation response has a success status and expected fields."""
    assert resp.status_code in (200, 201), (
        f"Expected 200/201 for API key creation, got {resp.status_code}: {resp.text[:200]}"
    )
    for field in required_fields:
        assert field in body, f"Response must contain '{field}'"
