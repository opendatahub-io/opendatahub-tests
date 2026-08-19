from __future__ import annotations

import json
from typing import Any, TypedDict
from urllib.parse import quote

import requests
import structlog
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import NotFoundError, ResourceNotFoundError
from requests import Response
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_gateway.models_as_a_service.utils import build_maas_headers
from utilities.resources.auth_policy import AuthPolicy

LOGGER = structlog.get_logger(name=__name__)

MAAS_GATEWAY_AUTH_POLICY_NAME = "maas-gateway-auth"

MAAS_AUTH_POLICY_FIXTURE_NAMES = (
    "external_model_auth_policy",
    "maas_auth_policy_tinyllama_premium",
    "maas_auth_policy_tinyllama_free",
    "oidc_auth_policy_patched",
)


DEFAULT_MAAS_TENANT: str = "models-as-a-service"


class FreeUserKeysAcrossSubscriptions(TypedDict):
    """API keys minted for the free user across two MaaS subscriptions."""

    username: str
    primary_subscription_name: str
    secondary_subscription_name: str
    primary_subscription_key_ids: list[str]
    secondary_subscription_key_ids: list[str]


def assert_tenant_field(body: dict[str, Any], context: str, expected: str = DEFAULT_MAAS_TENANT) -> None:
    """Assert that the response body contains a 'tenant' field with the expected value.

    Args:
        body: Parsed JSON response body from a MaaS API key endpoint.
        context: Human-readable label for assertion error messages (e.g. "GET /v1/api-keys/{id}").
        expected: Expected tenant value. Defaults to the product default tenant namespace.
    """
    assert "tenant" in body, f"Expected 'tenant' field in {context} response, got keys: {list(body.keys())}"
    assert body["tenant"] == expected, f"Expected tenant={expected!r} in {context} response, got: {body['tenant']!r}"


def assert_key_rejected_at_inference(
    request_session_http: requests.Session,
    inference_url: str,
    plaintext_key: str,
    payload: dict[str, Any],
    expected_status: int = 403,
    wait_timeout: int = 60,
    sleep: int = 2,
) -> None:
    """Poll inference endpoint until the API key is rejected with expected status."""
    headers = build_maas_headers(token=plaintext_key)
    for response in TimeoutSampler(
        wait_timeout=wait_timeout,
        sleep=sleep,
        func=request_session_http.post,
        url=inference_url,
        headers=headers,
        json=payload,
        timeout=10,
    ):
        LOGGER.info(f"Polling inference: status={response.status_code} expected={expected_status}")
        if response.status_code == expected_status:
            break

    assert response.status_code == expected_status, (
        f"Expected {expected_status}, got {response.status_code}: {(response.text or '')[:200]}"
    )


def assert_key_rejected_on_endpoint(
    request_session_http: requests.Session,
    url: str,
    plaintext_key: str,
    expected_status: int = 403,
    wait_timeout: int = 30,
    sleep: int = 2,
) -> None:
    """Poll a GET endpoint until the API key is rejected with expected status."""
    headers = build_maas_headers(token=plaintext_key)
    for response in TimeoutSampler(
        wait_timeout=wait_timeout,
        sleep=sleep,
        func=request_session_http.get,
        url=url,
        headers=headers,
        timeout=10,
    ):
        LOGGER.info(f"Polling endpoint: status={response.status_code} expected={expected_status}")
        if response.status_code == expected_status:
            break

    assert response.status_code == expected_status, (
        f"Expected {expected_status}, got {response.status_code}: {(response.text or '')[:200]}"
    )


def build_chat_payload(model_name: str, prompt: str = "hello", max_tokens: int = 1) -> dict[str, Any]:
    """Build a minimal chat completions request payload."""
    return {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }


def get_api_key(
    request_session_http: requests.Session,
    base_url: str,
    key_id: str,
    ocp_user_token: str,
    request_timeout_seconds: int = 60,
    extra_headers: dict[str, str] | None = None,
) -> tuple[Response, dict[str, Any]]:
    """Fetch a single API key by ID via MaaS API (GET /v1/api-keys/{id})."""
    url = f"{base_url}/v1/api-keys/{quote(key_id, safe='')}"
    request_headers = build_maas_headers(token=ocp_user_token)
    if extra_headers is not None:
        request_headers.update(extra_headers)
    response = request_session_http.get(
        url=url,
        headers=request_headers,
        timeout=request_timeout_seconds,
    )
    LOGGER.info(f"get_api_key: url={url} key_id={key_id} status={response.status_code}")
    try:
        parsed_body: dict[str, Any] = json.loads(response.text)
    except json.JSONDecodeError as error:
        raise AssertionError(
            f"get_api_key returned non-JSON response: status={response.status_code} body={response.text[:200]}"
        ) from error
    return response, parsed_body


def list_api_keys(
    request_session_http: requests.Session,
    base_url: str,
    ocp_user_token: str,
    filters: dict[str, Any] | None = None,
    sort: dict[str, Any] | None = None,
    pagination: dict[str, Any] | None = None,
    request_timeout_seconds: int = 60,
    extra_headers: dict[str, str] | None = None,
) -> tuple[Response, dict[str, Any]]:
    """Search/list API keys via MaaS API (POST /v1/api-keys/search)."""
    url = f"{base_url}/v1/api-keys/search"
    payload: dict[str, Any] = {}
    if filters is not None:
        payload["filters"] = filters
    if sort is not None:
        payload["sort"] = sort
    if pagination is not None:
        payload["pagination"] = pagination

    request_headers = build_maas_headers(token=ocp_user_token)
    if extra_headers is not None:
        request_headers.update(extra_headers)
    response = request_session_http.post(
        url=url,
        headers=request_headers,
        json=payload,
        timeout=request_timeout_seconds,
    )
    LOGGER.info(f"list_api_keys: url={url} status={response.status_code}")
    try:
        parsed_body: dict[str, Any] = json.loads(response.text)
    except json.JSONDecodeError as error:
        raise AssertionError(
            f"list_api_keys returned non-JSON response: status={response.status_code} body={response.text[:200]}"
        ) from error
    return response, parsed_body


def resolve_api_key_username(
    request_session_http: requests.Session,
    base_url: str,
    key_id: str,
    ocp_user_token: str,
) -> str:
    """Fetch an API key by ID and return the owner's username."""
    get_resp, get_body = get_api_key(
        request_session_http=request_session_http,
        base_url=base_url,
        key_id=key_id,
        ocp_user_token=ocp_user_token,
    )
    assert get_resp.status_code == 200, (
        f"Expected 200 on GET /v1/api-keys/{key_id}, got {get_resp.status_code}: {get_resp.text[:200]}"
    )
    username = get_body.get("username") or get_body.get("owner")
    assert username, "Expected 'username' or 'owner' field in GET response"
    return username


def bulk_revoke_api_keys(
    request_session_http: requests.Session,
    base_url: str,
    ocp_user_token: str,
    username: str | None = None,
    subscription: str | None = None,
    dry_run: bool | None = None,
    request_timeout_seconds: int = 60,
) -> tuple[Response, dict[str, Any]]:
    """Bulk revoke API keys via MaaS API (POST /v1/api-keys/bulk-revoke).

    Args:
        request_session_http: HTTP session for the request.
        base_url: MaaS API base URL.
        ocp_user_token: OCP bearer token for the actor performing the revoke.
        username: Optional owner username scope for the bulk revoke.
        subscription: Optional MaaSSubscription name scope for the bulk revoke.
        dry_run: When True, preview matching keys without revoking them.
        request_timeout_seconds: Request timeout in seconds.
    """
    url = f"{base_url}/v1/api-keys/bulk-revoke"
    payload: dict[str, Any] = {}
    if username is not None:
        payload["username"] = username
    if subscription is not None:
        payload["subscription"] = subscription
    if dry_run is not None:
        payload["dryRun"] = dry_run
    response = request_session_http.post(
        url=url,
        headers={"Authorization": f"Bearer {ocp_user_token}", "Content-Type": "application/json"},
        json=payload,
        timeout=request_timeout_seconds,
    )
    LOGGER.info(
        f"bulk_revoke_api_keys: url={url} username={username} subscription={subscription} "
        f"dry_run={dry_run} status={response.status_code}"
    )
    try:
        parsed_body: dict[str, Any] = json.loads(response.text)
    except json.JSONDecodeError as error:
        raise AssertionError(
            f"bulk_revoke_api_keys returned non-JSON response: status={response.status_code} body={response.text[:200]}"
        ) from error
    return response, parsed_body


def revoked_count_from_bulk_body(bulk_body: dict[str, Any]) -> int:
    """Return revokedCount from a bulk-revoke response body."""
    if "revokedCount" not in bulk_body:
        return 0
    return int(bulk_body["revokedCount"])


def assert_api_key_status(
    request_session_http: requests.Session,
    base_url: str,
    key_id: str,
    ocp_user_token: str,
    expected_status: str,
) -> None:
    """Assert a single API key has the expected status via GET /v1/api-keys/{id}."""
    get_resp, get_body = get_api_key(
        request_session_http=request_session_http,
        base_url=base_url,
        key_id=key_id,
        ocp_user_token=ocp_user_token,
    )
    assert get_resp.status_code == 200, (
        f"Expected 200 on GET /v1/api-keys/{key_id}, got {get_resp.status_code}: {get_resp.text[:200]}"
    )
    assert "status" in get_body, (
        f"Expected 'status' in GET response for key id={key_id}, got keys: {list(get_body.keys())}"
    )
    assert get_body["status"] == expected_status, (
        f"Expected key id={key_id} to have status={expected_status!r}, got: {get_body['status']!r}"
    )


def assert_api_keys_status(
    request_session_http: requests.Session,
    base_url: str,
    key_ids: list[str],
    ocp_user_token: str,
    expected_status: str,
) -> None:
    """Assert each API key in key_ids has the expected status."""
    for key_id in key_ids:
        assert_api_key_status(
            request_session_http=request_session_http,
            base_url=base_url,
            key_id=key_id,
            ocp_user_token=ocp_user_token,
            expected_status=expected_status,
        )


def assert_bulk_revoke_success(
    request_session_http: requests.Session,
    base_url: str,
    ocp_user_token: str,
    username: str,
    min_revoked_count: int = 1,
    subscription: str | None = None,
) -> int:
    """Bulk revoke API keys and assert the operation succeeded."""
    bulk_resp, bulk_body = bulk_revoke_api_keys(
        request_session_http=request_session_http,
        base_url=base_url,
        ocp_user_token=ocp_user_token,
        username=username,
        subscription=subscription,
    )
    assert bulk_resp.status_code == 200, (
        f"Expected 200 on bulk-revoke for user {username}, got {bulk_resp.status_code}: {bulk_resp.text[:200]}"
    )
    revoked_count = revoked_count_from_bulk_body(bulk_body=bulk_body)
    assert revoked_count >= min_revoked_count, (
        f"Expected at least {min_revoked_count} revoked key(s), got revokedCount={revoked_count}"
    )
    return revoked_count


def assert_bulk_revoke_by_subscription_success(
    request_session_http: requests.Session,
    base_url: str,
    ocp_user_token: str,
    subscription: str,
    min_revoked_count: int = 1,
) -> int:
    """Bulk revoke by subscription scope only and assert the operation succeeded."""
    bulk_resp, bulk_body = bulk_revoke_api_keys(
        request_session_http=request_session_http,
        base_url=base_url,
        ocp_user_token=ocp_user_token,
        subscription=subscription,
    )
    assert bulk_resp.status_code == 200, (
        f"Expected 200 on subscription-scoped bulk-revoke for subscription={subscription!r}, "
        f"got {bulk_resp.status_code}: {bulk_resp.text[:200]}"
    )
    revoked_count = revoked_count_from_bulk_body(bulk_body=bulk_body)
    assert revoked_count >= min_revoked_count, (
        f"Expected at least {min_revoked_count} revoked key(s) for subscription={subscription!r}, "
        f"got revokedCount={revoked_count}"
    )
    return revoked_count


def assert_bulk_dry_run_preview(
    request_session_http: requests.Session,
    base_url: str,
    ocp_user_token: str,
    expected_revoked_count: int,
    username: str | None = None,
    subscription: str | None = None,
) -> int:
    """Assert bulk-revoke dry-run returns dryRun=true and expected revokedCount without mutating keys."""
    bulk_resp, bulk_body = bulk_revoke_api_keys(
        request_session_http=request_session_http,
        base_url=base_url,
        ocp_user_token=ocp_user_token,
        username=username,
        subscription=subscription,
        dry_run=True,
    )
    assert bulk_resp.status_code == 200, (
        f"Expected 200 on bulk-revoke dry-run, got {bulk_resp.status_code}: {bulk_resp.text[:200]}"
    )
    assert "dryRun" in bulk_body, f"Expected 'dryRun' in bulk-revoke response, got keys: {list(bulk_body.keys())}"
    assert bulk_body["dryRun"] is True, f"Expected dryRun=true in bulk-revoke response, got: {bulk_body['dryRun']!r}"
    revoked_count = revoked_count_from_bulk_body(bulk_body=bulk_body)
    assert revoked_count == expected_revoked_count, (
        f"Expected dry-run revokedCount={expected_revoked_count}, got revokedCount={revoked_count}"
    )
    return revoked_count


def assert_api_key_get_ok(resp: Response, body: dict[str, Any], key_id: str) -> None:
    """Assert a GET /v1/api-keys/{id} response has status 200."""
    assert resp.status_code == 200, (
        f"Expected 200 on GET /v1/api-keys/{key_id}, got {resp.status_code}: {resp.text[:200]}"
    )


def search_active_api_keys(
    request_session_http: requests.Session,
    base_url: str,
    ocp_user_token: str,
    include_ephemeral: bool = False,
    request_timeout_seconds: int = 30,
) -> list[dict[str, Any]]:
    """POST /v1/api-keys/search for active keys and return the list of matching items."""
    filters: dict[str, Any] = {"status": ["active"]}
    if include_ephemeral:
        filters["includeEphemeral"] = True
    url = f"{base_url}/v1/api-keys/search"
    resp = request_session_http.post(
        url=url,
        headers={"Authorization": f"Bearer {ocp_user_token}"},
        json={"filters": filters, "pagination": {"limit": 50, "offset": 0}},
        timeout=request_timeout_seconds,
    )
    assert resp.status_code == 200, f"Expected 200 from key search, got {resp.status_code}: {(resp.text or '')[:200]}"
    body = resp.json()
    return body.get("items") or body.get("data") or []


def build_inference_url(maas_scheme: str, maas_host: str, model_name: str) -> str:
    """Build the chat completions inference URL for a given model."""
    return f"{maas_scheme}://{maas_host}/llm/{model_name}/v1/chat/completions"


def _mapping_get(mapping: Any, key: str) -> Any:
    """Read a key from a Kubernetes API object or plain dict."""
    if mapping is None:
        return None
    if isinstance(mapping, dict):
        return mapping.get(key)
    return getattr(mapping, key, None)


def _api_key_validation_callback_url_from_rules(rules: Any) -> str | None:
    """Return apiKeyValidation.http.url from an AuthPolicy rules block, if present."""
    metadata = _mapping_get(mapping=rules, key="metadata")
    api_key_validation = _mapping_get(mapping=metadata, key="apiKeyValidation")
    http_config = _mapping_get(mapping=api_key_validation, key="http")
    callback_url = _mapping_get(mapping=http_config, key="url")
    return str(callback_url) if callback_url else None


def get_auth_policy_condition(
    admin_client: DynamicClient,
    policy_name: str,
    namespace: str,
    condition_type: str,
) -> dict[str, Any] | None:
    """Find a specific condition by type from an AuthPolicy's status."""
    auth_policy = AuthPolicy(
        client=admin_client,
        name=policy_name,
        namespace=namespace,
    )
    assert auth_policy.exists, f"AuthPolicy '{policy_name}' not found in namespace '{namespace}'"
    conditions: list[dict[str, Any]] = (auth_policy.instance.status or {}).get("conditions") or []
    return next(
        (condition for condition in conditions if condition.get("type") == condition_type),
        None,
    )


def wait_for_auth_policy_accepted(
    admin_client: DynamicClient,
    policy_name: str,
    namespace: str,
    timeout: int = 300,
    reconciliation_hint: str = ("Ensure a MaaSAuthPolicy exists to trigger gateway auth reconciliation."),
) -> AuthPolicy:
    """Poll until an AuthPolicy exists and Accepted and Enforced conditions are True.

    Accepted alone is not enough for ExtAuth: Authorino may still be reconciling, so
    unauthenticated /maas-api calls can return 200 and API key create can fail with
    AUTH_FAILURE (missing X-MaaS-Username). Wait for Enforced before probing the API.
    """
    auth_policy = AuthPolicy(
        client=admin_client,
        name=policy_name,
        namespace=namespace,
    )
    try:
        for _ in TimeoutSampler(
            wait_timeout=timeout,
            sleep=5,
            func=auth_policy.get,
            exceptions_dict={NotFoundError: [], ResourceNotFoundError: []},
        ):
            if not auth_policy.exists:
                continue
            accepted_condition = get_auth_policy_condition(
                admin_client=admin_client,
                policy_name=policy_name,
                namespace=namespace,
                condition_type="Accepted",
            )
            enforced_condition = get_auth_policy_condition(
                admin_client=admin_client,
                policy_name=policy_name,
                namespace=namespace,
                condition_type="Enforced",
            )
            accepted = accepted_condition is not None and accepted_condition.get("status") == "True"
            enforced = enforced_condition is not None and enforced_condition.get("status") == "True"
            if accepted and enforced:
                LOGGER.info(
                    f"AuthPolicy '{namespace}/{policy_name}' is Accepted and Enforced "
                    "after MaaSAuthPolicy reconciliation"
                )
                return auth_policy
    except TimeoutExpiredError as error:
        raise AssertionError(
            f"Timed out waiting for AuthPolicy '{namespace}/{policy_name}' to become "
            f"Accepted and Enforced. {reconciliation_hint}"
        ) from error
    raise AssertionError(f"AuthPolicy '{namespace}/{policy_name}' did not become Accepted and Enforced")


def get_auth_policy_callback_url(
    admin_client: DynamicClient,
    policy_name: str,
    namespace: str,
) -> str:
    """Read the apiKeyValidation callback URL from a MaaS AuthPolicy."""
    auth_policy = AuthPolicy(
        client=admin_client,
        name=policy_name,
        namespace=namespace,
        ensure_exists=True,
    )
    spec = auth_policy.instance.spec
    rules_blocks = (
        _mapping_get(mapping=_mapping_get(mapping=spec, key="defaults"), key="rules"),
        _mapping_get(mapping=spec, key="rules"),
        _mapping_get(mapping=_mapping_get(mapping=spec, key="overrides"), key="rules"),
    )
    for rules in rules_blocks:
        callback_url = _api_key_validation_callback_url_from_rules(rules=rules)
        if callback_url:
            LOGGER.info(f"get_auth_policy_callback_url: policy='{policy_name}' url='{callback_url}'")
            return callback_url

    configured_blocks = [
        block_name
        for block_name, rules in (
            ("spec.defaults.rules", rules_blocks[0]),
            ("spec.rules", rules_blocks[1]),
            ("spec.overrides.rules", rules_blocks[2]),
        )
        if rules is not None
    ]
    raise AssertionError(
        f"AuthPolicy '{policy_name}' in namespace '{namespace}' has no "
        f"metadata.apiKeyValidation.http.url. "
        f"Found rules blocks: {configured_blocks or ['none']}"
    )
