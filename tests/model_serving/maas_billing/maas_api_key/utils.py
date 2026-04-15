from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote

import requests
import structlog
from kubernetes.dynamic import DynamicClient
from requests import Response

from utilities.resources.auth_policy import AuthPolicy

LOGGER = structlog.get_logger(name=__name__)


def get_api_key(
    request_session_http: requests.Session,
    base_url: str,
    key_id: str,
    ocp_user_token: str,
    request_timeout_seconds: int = 60,
) -> tuple[Response, dict[str, Any]]:
    """Fetch a single API key by ID via MaaS API (GET /v1/api-keys/{id})."""
    url = f"{base_url}/v1/api-keys/{quote(key_id, safe='')}"
    response = request_session_http.get(
        url=url,
        headers={"Authorization": f"Bearer {ocp_user_token}"},
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

    response = request_session_http.post(
        url=url,
        headers={"Authorization": f"Bearer {ocp_user_token}"},
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
    username: str,
    request_timeout_seconds: int = 60,
) -> tuple[Response, dict[str, Any]]:
    """Bulk revoke all active API keys for a given user via MaaS API (POST /v1/api-keys/bulk-revoke)."""
    url = f"{base_url}/v1/api-keys/bulk-revoke"
    response = request_session_http.post(
        url=url,
        headers={"Authorization": f"Bearer {ocp_user_token}", "Content-Type": "application/json"},
        json={"username": username},
        timeout=request_timeout_seconds,
    )
    LOGGER.info(f"bulk_revoke_api_keys: url={url} username={username} status={response.status_code}")
    try:
        parsed_body: dict[str, Any] = json.loads(response.text)
    except json.JSONDecodeError as error:
        raise AssertionError(
            f"bulk_revoke_api_keys returned non-JSON response: status={response.status_code} body={response.text[:200]}"
        ) from error
    return response, parsed_body


def assert_bulk_revoke_success(
    request_session_http: requests.Session,
    base_url: str,
    ocp_user_token: str,
    username: str,
    min_revoked_count: int = 1,
) -> int:
    """Bulk revoke API keys for a user and assert the operation succeeded."""
    bulk_resp, bulk_body = bulk_revoke_api_keys(
        request_session_http=request_session_http,
        base_url=base_url,
        ocp_user_token=ocp_user_token,
        username=username,
    )
    assert bulk_resp.status_code == 200, (
        f"Expected 200 on bulk-revoke for user {username}, got {bulk_resp.status_code}: {bulk_resp.text[:200]}"
    )
    revoked_count: int = bulk_body.get("revokedCount", 0)
    assert revoked_count >= min_revoked_count, (
        f"Expected at least {min_revoked_count} revoked key(s), got revokedCount={revoked_count}"
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
    try:
        callback_url: str = auth_policy.instance.spec.rules.metadata.apiKeyValidation.http.url
    except AttributeError as error:
        raise AssertionError(
            f"AuthPolicy '{policy_name}' in namespace '{namespace}' is missing "
            f"the apiKeyValidation callback URL field: {error}"
        ) from error
    LOGGER.info(f"get_auth_policy_callback_url: policy='{policy_name}' url='{callback_url}'")
    return callback_url


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
