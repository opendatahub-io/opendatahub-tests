from __future__ import annotations

import base64
import json
from collections.abc import Generator
from typing import Any

import requests
import structlog
from kubernetes.dynamic import DynamicClient

from tests.model_serving.maas_billing.maas_subscription.utils import create_maas_subscription
from utilities.general import generate_random_name
from utilities.user_utils import get_byoidc_issuer_url

LOGGER = structlog.get_logger(name=__name__)

MAAS_API_AUTH_POLICY_NAME = "maas-api-auth-policy"
MAAS_OIDC_REALM = "openshift-ai-maas"
MAAS_OIDC_GROUP = "maas-users"
OIDC_CLIENT_ID = "maas-client"
JWT_WHEN_PREDICATE = (
    '!request.headers.authorization.startsWith("Bearer sk-oai-") && '
    'request.headers.authorization.matches("^Bearer [^.]+\\\\.[^.]+\\\\.[^.]+$")'
)


def get_maas_oidc_issuer_url(admin_client: DynamicClient, realm: str = MAAS_OIDC_REALM) -> str:
    """Derive a realm-specific Keycloak issuer URL from the cluster's byoidc OIDC provider."""
    byoidc_url = get_byoidc_issuer_url(admin_client=admin_client)
    keycloak_base, _, _old_realm = byoidc_url.rpartition("/realms/")
    assert keycloak_base, f"Could not extract Keycloak base from byoidc issuer URL: {byoidc_url}"
    return f"{keycloak_base}/realms/{realm}"


def request_oidc_access_token(
    request_session_http: requests.Session,
    token_url: str,
    client_id: str,
    username: str,
    password: str,
    client_secret: str | None = None,
    request_timeout_seconds: int = 30,
) -> str:
    """Exchange user credentials for an OIDC access token via the password grant."""
    token_data: dict[str, str] = {
        "grant_type": "password",
        "client_id": client_id,
        "username": username,
        "password": password,
        "scope": "openid",
    }
    if client_secret:
        token_data["client_secret"] = client_secret
    response = request_session_http.post(
        url=token_url,
        data=token_data,
        timeout=request_timeout_seconds,
    )
    assert response.status_code == 200, (
        f"OIDC token request failed: status={response.status_code} body={response.text[:300]}"
    )

    access_token: str | None = response.json().get("access_token")
    assert access_token, "OIDC token response missing 'access_token'"

    LOGGER.info(f"request_oidc_access_token: acquired token for user '{username}' (length={len(access_token)})")
    return access_token


def request_oidc_token_raw(
    request_session_http: requests.Session,
    token_url: str,
    client_id: str,
    username: str,
    password: str,
    client_secret: str | None = None,
    request_timeout_seconds: int = 30,
) -> requests.Response:
    """Send a token request and return the raw response without assertions.

    Useful for negative tests (wrong password, nonexistent user) where
    non-200 responses are expected.
    """
    token_data: dict[str, str] = {
        "grant_type": "password",
        "client_id": client_id,
        "username": username,
        "password": password,
        "scope": "openid",
    }
    if client_secret:
        token_data["client_secret"] = client_secret
    response = request_session_http.post(
        url=token_url,
        data=token_data,
        timeout=request_timeout_seconds,
    )
    LOGGER.info(f"request_oidc_token_raw: user='{username}' status={response.status_code}")
    return response


def get_active_key_ids(
    request_session_http: requests.Session,
    base_url: str,
    ocp_user_token: str,
) -> list[str]:
    """List active API key IDs for the given user token."""
    from tests.model_serving.maas_billing.maas_api_key.utils import list_api_keys

    list_response, list_body = list_api_keys(
        request_session_http=request_session_http,
        base_url=base_url,
        ocp_user_token=ocp_user_token,
        filters={"status": ["active"]},
        sort={"by": "created_at", "order": "desc"},
        pagination={"limit": 50, "offset": 0},
    )
    assert list_response.status_code == 200, (
        f"list_api_keys failed: {list_response.status_code}: {list_response.text[:200]}"
    )

    items = list_body.get("items") or list_body.get("data") or []
    return [key_entry["id"] for key_entry in items]


def decode_jwt_payload(token: str) -> dict[str, object]:
    """Decode the payload (second segment) of a JWT without signature verification."""
    parts = token.split(".")
    assert len(parts) == 3, f"Expected 3 JWT segments, got {len(parts)}"

    payload_segment = parts[1]
    padding = "=" * (4 - len(payload_segment) % 4)
    payload: dict[str, object] = json.loads(base64.urlsafe_b64decode(s=payload_segment + padding))

    LOGGER.info(f"decode_jwt_payload: decoded JWT with keys={list(payload.keys())}")
    return payload


def fetch_models_with_header(
    session: requests.Session,
    models_url: str,
    api_key: str,
    extra_headers: dict[str, str] | None = None,
) -> requests.Response:
    """GET /v1/models with API key auth and optional extra headers."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    return session.get(url=models_url, headers=headers, timeout=60)


def fetch_models_with_spoofed_header(
    session: requests.Session,
    base_url: str,
    api_key: str,
    extra_headers: dict[str, str] | None = None,
) -> requests.Response:
    """GET /v1/models with API key auth and optional spoofed identity headers."""
    return fetch_models_with_header(
        session=session,
        models_url=f"{base_url}/v1/models",
        api_key=api_key,
        extra_headers=extra_headers,
    )


def assert_model_lists_match(
    baseline_response: requests.Response,
    spoofed_response: requests.Response,
    injection_description: str,
) -> None:
    """Assert that two /v1/models responses return identical model IDs.

    Used by header injection security tests to verify that injected headers
    do not change the model list returned by the gateway.
    """
    baseline_models = baseline_response.json().get("data", [])
    spoofed_models = spoofed_response.json().get("data", [])
    baseline_model_ids = sorted(model["id"] for model in baseline_models)
    spoofed_model_ids = sorted(model["id"] for model in spoofed_models)
    assert spoofed_model_ids == baseline_model_ids, (
        f"Injected {injection_description} changed model list (possible escalation). "
        f"Baseline: {baseline_model_ids}, spoofed: {spoofed_model_ids}"
    )


def create_oidc_subscription(
    admin_client: DynamicClient,
    subscription_namespace: str,
    model_name: str,
    model_namespace: str,
) -> Generator[Any, Any, Any]:
    """Create a MaaSModelRef + MaaSSubscription owned by the MaaS OIDC group."""
    from ocp_resources.maas_model_ref import MaaSModelRef

    sub_name = f"e2e-oidc-sub-{generate_random_name()}"

    with (
        MaaSModelRef(
            client=admin_client,
            name=model_name,
            namespace=model_namespace,
            model_ref={"name": model_name, "namespace": model_namespace, "kind": "LLMInferenceService"},
            teardown=True,
            wait_for_resource=True,
        ) as model_ref,
        create_maas_subscription(
            admin_client=admin_client,
            subscription_namespace=subscription_namespace,
            subscription_name=sub_name,
            owner_group_name=MAAS_OIDC_GROUP,
            model_name=model_ref.name,
            model_namespace=model_ref.namespace,
            tokens_per_minute=1000,
            window="1m",
            priority=0,
            teardown=True,
            wait_for_resource=True,
        ) as subscription,
    ):
        yield subscription
