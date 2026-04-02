from __future__ import annotations

import json
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import Any
from urllib.parse import quote, urlparse

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.llm_inference_service import LLMInferenceService
from ocp_resources.maas_subscription import MaaSSubscription
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from requests import Response
from timeout_sampler import TimeoutSampler

from utilities.constants import (
    MAAS_GATEWAY_NAME,
    MAAS_GATEWAY_NAMESPACE,
    ApiGroups,
)
from utilities.general import generate_random_name
from utilities.resources.auth import Auth

LOGGER = structlog.get_logger(name=__name__)
MAAS_SUBSCRIPTION_NAMESPACE = "models-as-a-service"
MAAS_DB_NAMESPACE = "redhat-ods-applications"
POSTGRES_DEPLOYMENT_NAME = "postgres"
POSTGRES_SERVICE_NAME = "postgres"
POSTGRES_CREDS_SECRET_NAME = "postgres-creds"  # pragma: allowlist secret
MAAS_DB_CONFIG_SECRET_NAME = "maas-db-config"  # pragma: allowlist secret
POSTGRES_IMAGE = "registry.redhat.io/rhel9/postgresql-15:latest"
POSTGRES_READY_LOG_TEXT = "accepting connections"


@contextmanager
def patch_llmisvc_with_maas_router_and_tiers(
    llm_service: LLMInferenceService,
    tiers: Sequence[str],
    enable_auth: bool = True,
) -> Generator[None]:
    """
    Patch an LLMInferenceService to use MaaS router (gateway refs + route {})
    and set MaaS tier annotation.

    This is intended for MaaS subscription tests where you want distinct
    tiered models (e.g. free vs premium)

    Examples:
      - tiers=[]              -> open model
      - tiers=["premium"]     -> premium-only
    """
    router_spec = {
        "gateway": {"refs": [{"name": MAAS_GATEWAY_NAME, "namespace": MAAS_GATEWAY_NAMESPACE}]},
        "route": {},
    }

    tiers_val = list(tiers)
    patch_body = {
        "metadata": {
            "annotations": {
                f"alpha.{ApiGroups.MAAS_IO}/tiers": json.dumps(tiers_val),
                "security.opendatahub.io/enable-auth": "true" if enable_auth else "false",
            }
        },
        "spec": {"router": router_spec},
    }

    with ResourceEditor(patches={llm_service: patch_body}):
        yield


def model_id_from_chat_completions_url(model_url: str) -> str:
    path = urlparse(model_url).path.strip("/")
    parts = path.split("/")

    if len(parts) >= 2 and parts[0] == "llm":
        model_id = parts[1]
        if model_id:
            return model_id

    raise AssertionError(f"Cannot extract model id from url: {model_url!r} (path={path!r})")


def chat_payload_for_url(model_url: str, *, prompt: str = "Hello", max_tokens: int = 8) -> dict:
    model_id = model_id_from_chat_completions_url(model_url=model_url)
    return {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }


def poll_expected_status(
    request_session_http: requests.Session,
    model_url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    expected_statuses: set[int],
    wait_timeout: int = 240,
    sleep: int = 5,
    request_timeout: int = 60,
) -> requests.Response:
    """
    Poll model endpoint until we see one of `expected_statuses` or timeout.

    Returns the response that matched expected status.
    """
    last_response: requests.Response | None = None
    observed_responses: list[tuple[int | None, str]] = []

    for response in TimeoutSampler(
        wait_timeout=wait_timeout,
        sleep=sleep,
        func=request_session_http.post,
        url=model_url,
        headers=headers,
        json=payload,
        timeout=request_timeout,
    ):
        last_response = response
        status_code = getattr(response, "status_code", None)
        response_text = (getattr(response, "text", "") or "")[:200]

        observed_responses.append((status_code, response_text))

        LOGGER.info(f"Polling model_url={model_url} status={status_code} expected={sorted(expected_statuses)}")

        if status_code in expected_statuses:
            return response

    pytest.fail(
        "Timed out waiting for expected HTTP status. "
        f"model_url={model_url}, "
        f"expected={sorted(expected_statuses)}, "
        f"last_status={getattr(last_response, 'status_code', None)}, "
        f"last_body={(getattr(last_response, 'text', '') or '')[:200]}, "
        f"seen_count={len(observed_responses)}"
    )


def create_maas_subscription(
    admin_client: DynamicClient,
    subscription_namespace: str,
    subscription_name: str,
    owner_group_name: str,
    model_name: str,
    model_namespace: str,
    tokens_per_minute: int,
    window: str = "1m",
    priority: int = 0,
    teardown: bool = True,
    wait_for_resource: bool = True,
) -> MaaSSubscription:

    return MaaSSubscription(
        client=admin_client,
        name=subscription_name,
        namespace=subscription_namespace,
        owner={
            "groups": [{"name": owner_group_name}],
        },
        model_refs=[
            {
                "name": model_name,
                "namespace": model_namespace,
                "tokenRateLimits": [{"limit": tokens_per_minute, "window": window}],
            }
        ],
        priority=priority,
        teardown=teardown,
        wait_for_resource=wait_for_resource,
    )


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
    """
    Create an API key via MaaS API and return (response, parsed_body).

    Uses ocp_user_token for auth against maas-api.
    Expects plaintext key in body["key"] (sk-...).

    Args:
        expires_in: Optional expiration duration string (e.g. "24h", "720h").
            When None, no expiresIn field is sent and the key does not expire.
        raise_on_error: When True (default), raises AssertionError for non-200/201
            responses. Set to False when testing error cases (e.g. 400 rejection).
        subscription: Optional MaaSSubscription name to bind at mint time.
            When provided, the key is bound to this subscription for inference.
            When None, the API auto-selects the highest-priority subscription.
        ephemeral: When True, marks the key as short-lived/programmatic.
            Ephemeral keys are hidden from default search results and are
            cleaned up automatically by the cleanup CronJob after expiration.
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
        headers={
            "Authorization": f"Bearer {ocp_user_token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=request_timeout_seconds,
    )

    LOGGER.info(f"create_api_key: url={api_keys_url} status={response.status_code}")
    if response.status_code not in (200, 201):
        if raise_on_error:
            raise AssertionError(f"api-key create failed: status={response.status_code}")
        return response, {}

    try:
        parsed_body: dict[str, Any] = json.loads(response.text)
    except json.JSONDecodeError as error:
        LOGGER.error(f"Unable to parse API key response from {api_keys_url}; status={response.status_code}")
        raise AssertionError("API key creation returned non-JSON response") from error

    api_key = parsed_body.get("key", "")
    if not isinstance(api_key, str) or not api_key.startswith("sk-"):
        raise AssertionError("No plaintext api key returned in MaaS API response")

    return response, parsed_body


def get_api_key(
    request_session_http: requests.Session,
    base_url: str,
    key_id: str,
    ocp_user_token: str,
    request_timeout_seconds: int = 60,
) -> tuple[Response, dict[str, Any]]:
    """
    Fetch a single API key by ID via MaaS API (GET /v1/api-keys/{id}).
    """
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
    """
    Search/list API keys via MaaS API (POST /v1/api-keys/search).
    """
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
    LOGGER.info(f"list_api_keys: url={url} status={response.status_code} items_count=pending_parse")
    try:
        parsed_body: dict[str, Any] = json.loads(response.text)
    except json.JSONDecodeError as error:
        raise AssertionError(
            f"list_api_keys returned non-JSON response: status={response.status_code} body={response.text[:200]}"
        ) from error
    return response, parsed_body


def wait_for_auth_ready(auth: Auth, baseline_time: str, timeout: int = 60) -> None:
    """Wait for Auth CR to reconcile after a patch."""
    for instance in TimeoutSampler(wait_timeout=timeout, sleep=2, func=lambda: auth.instance):
        auth_conditions = (instance.status or {}).get("conditions") or []
        ready_condition = next(
            (condition for condition in auth_conditions if condition.get("type") == "Ready"),
            None,
        )
        if (
            ready_condition
            and ready_condition.get("lastTransitionTime") != baseline_time
            and ready_condition.get("status") == "True"
        ):
            return


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


def create_and_yield_api_key_id(
    request_session_http: requests.Session,
    base_url: str,
    ocp_user_token: str,
    key_name_prefix: str,
    expires_in: str | None = None,
) -> Generator[str]:
    """Create an API key, yield its ID, and revoke it on teardown."""
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


def revoke_api_key(
    request_session_http: requests.Session,
    base_url: str,
    key_id: str,
    ocp_user_token: str,
    request_timeout_seconds: int = 60,
) -> tuple[Response, dict[str, Any]]:
    """
    Revoke an API key via MaaS API (DELETE /v1/api-keys/{id}).
    """
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


def bulk_revoke_api_keys(
    request_session_http: requests.Session,
    base_url: str,
    ocp_user_token: str,
    username: str,
    request_timeout_seconds: int = 60,
) -> tuple[Response, dict[str, Any]]:
    """
    Bulk revoke all active API keys for a given user via MaaS API (POST /v1/api-keys/bulk-revoke).

    """
    url = f"{base_url}/v1/api-keys/bulk-revoke"
    response = request_session_http.post(
        url=url,
        headers={
            "Authorization": f"Bearer {ocp_user_token}",
            "Content-Type": "application/json",
        },
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


def assert_api_key_get_ok(resp: Response, body: dict[str, Any], key_id: str) -> None:
    """Assert a GET /v1/api-keys/{id} response has status 200."""
    assert resp.status_code == 200, (
        f"Expected 200 on GET /v1/api-keys/{key_id}, got {resp.status_code}: {resp.text[:200]}"
    )


def get_maas_postgres_labels() -> dict[str, str]:
    return {
        "app": "postgres",
        "purpose": "poc",
    }


def get_maas_api_labels() -> dict[str, str]:
    return {
        "app": "maas-api",
        "purpose": "poc",
    }


def get_maas_postgres_secret_objects(
    client: DynamicClient,
    namespace: str,
    teardown_resources: bool,
    postgres_user: str,
    postgres_password: str,
    postgres_db: str,
) -> list[Secret]:
    return [
        Secret(
            client=client,
            name=POSTGRES_CREDS_SECRET_NAME,
            namespace=namespace,
            string_data={
                "POSTGRES_USER": postgres_user,
                "POSTGRES_PASSWORD": postgres_password,
                "POSTGRES_DB": postgres_db,
            },
            label=get_maas_postgres_labels(),
            type="Opaque",
            teardown=teardown_resources,
        )
    ]


def get_maas_db_config_secret_objects(
    client: DynamicClient,
    namespace: str,
    teardown_resources: bool,
    postgres_user: str,
    postgres_password: str,
    postgres_db: str,
) -> list[Secret]:
    db_connection_url = (
        f"postgresql://{postgres_user}:{postgres_password}@{POSTGRES_SERVICE_NAME}:5432/{postgres_db}?sslmode=disable"
    )

    return [
        Secret(
            client=client,
            name=MAAS_DB_CONFIG_SECRET_NAME,
            namespace=namespace,
            string_data={"DB_CONNECTION_URL": db_connection_url},
            label=get_maas_api_labels(),
            type="Opaque",
            teardown=teardown_resources,
        )
    ]


def get_maas_postgres_service_objects(
    client: DynamicClient,
    namespace: str,
    teardown_resources: bool,
) -> list[Service]:
    return [
        Service(
            client=client,
            name=POSTGRES_SERVICE_NAME,
            namespace=namespace,
            selector={"app": "postgres"},
            ports=[
                {
                    "name": "postgres",
                    "port": 5432,
                    "protocol": "TCP",
                    "targetPort": 5432,
                }
            ],
            label=get_maas_postgres_labels(),
            teardown=teardown_resources,
        )
    ]


def get_maas_postgres_deployment_template_dict() -> dict[str, Any]:
    return {
        "metadata": {
            "labels": get_maas_postgres_labels(),
        },
        "spec": {
            "containers": [
                {
                    "name": "postgres",
                    "image": POSTGRES_IMAGE,
                    "env": [
                        {
                            "name": "POSTGRESQL_USER",
                            "valueFrom": {
                                "secretKeyRef": {
                                    "name": POSTGRES_CREDS_SECRET_NAME,
                                    "key": "POSTGRES_USER",
                                }
                            },
                        },
                        {
                            "name": "POSTGRESQL_PASSWORD",
                            "valueFrom": {
                                "secretKeyRef": {
                                    "name": POSTGRES_CREDS_SECRET_NAME,
                                    "key": "POSTGRES_PASSWORD",
                                }
                            },
                        },
                        {
                            "name": "POSTGRESQL_DATABASE",
                            "valueFrom": {
                                "secretKeyRef": {
                                    "name": POSTGRES_CREDS_SECRET_NAME,
                                    "key": "POSTGRES_DB",
                                }
                            },
                        },
                    ],
                    "ports": [{"containerPort": 5432}],
                    "volumeMounts": [{"name": "data", "mountPath": "/var/lib/pgsql/data"}],
                    "resources": {
                        "requests": {"memory": "256Mi", "cpu": "100m"},
                        "limits": {"memory": "512Mi", "cpu": "500m"},
                    },
                    "readinessProbe": {
                        "exec": {"command": ["/usr/libexec/check-container"]},
                        "initialDelaySeconds": 5,
                        "periodSeconds": 5,
                    },
                }
            ],
            "volumes": [{"name": "data", "emptyDir": {}}],
        },
    }


def get_maas_postgres_deployment_objects(
    client: DynamicClient,
    namespace: str,
    teardown_resources: bool,
) -> list[Deployment]:
    return [
        Deployment(
            client=client,
            name=POSTGRES_DEPLOYMENT_NAME,
            namespace=namespace,
            label=get_maas_postgres_labels(),
            replicas=1,
            selector={"matchLabels": {"app": "postgres"}},
            template=get_maas_postgres_deployment_template_dict(),
            wait_for_resource=True,
            teardown=teardown_resources,
        )
    ]


def get_maas_postgres_resources(
    client: DynamicClient,
    namespace: str,
    teardown_resources: bool,
    postgres_user: str,
    postgres_password: str,
    postgres_db: str,
) -> dict[Any, Any]:
    return {
        Secret: get_maas_postgres_secret_objects(
            client=client,
            namespace=namespace,
            teardown_resources=teardown_resources,
            postgres_user=postgres_user,
            postgres_password=postgres_password,
            postgres_db=postgres_db,
        )
        + get_maas_db_config_secret_objects(
            client=client,
            namespace=namespace,
            teardown_resources=teardown_resources,
            postgres_user=postgres_user,
            postgres_password=postgres_password,
            postgres_db=postgres_db,
        ),
        Service: get_maas_postgres_service_objects(
            client=client,
            namespace=namespace,
            teardown_resources=teardown_resources,
        ),
        Deployment: get_maas_postgres_deployment_objects(
            client=client,
            namespace=namespace,
            teardown_resources=teardown_resources,
        ),
    }


def wait_for_postgres_deployment_ready(
    admin_client: DynamicClient,
    namespace: str = MAAS_DB_NAMESPACE,
    timeout: int = 180,
) -> None:
    deployment = Deployment(
        client=admin_client,
        name=POSTGRES_DEPLOYMENT_NAME,
        namespace=namespace,
    )
    deployment.wait_for_condition(condition="Available", status="True", timeout=timeout)


def get_postgres_pod_in_namespace(
    admin_client: DynamicClient,
    namespace: str = MAAS_DB_NAMESPACE,
) -> Pod:
    postgres_pods = list(Pod.get(client=admin_client, namespace=namespace, label_selector="app=postgres"))
    assert postgres_pods, f"No PostgreSQL pod found in namespace {namespace}"
    return postgres_pods[0]


def wait_for_postgres_connection_log(
    admin_client: DynamicClient,
    namespace: str = MAAS_DB_NAMESPACE,
    timeout: int = 180,
    sleep: int = 5,
) -> None:
    for _ in TimeoutSampler(wait_timeout=timeout, sleep=sleep, func=lambda: True):
        postgres_pod = get_postgres_pod_in_namespace(admin_client=admin_client, namespace=namespace)
        pod_log = postgres_pod.log(container="postgres")
        if POSTGRES_READY_LOG_TEXT in pod_log:
            LOGGER.info(f"PostgreSQL pod is accepting connections in namespace {namespace}")
            return

    raise TimeoutError(f"PostgreSQL pod in namespace {namespace} did not report accepting connections")
