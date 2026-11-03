"""Loki and usage-logs tenancy-proxy query helpers for RHOAIENG-99404."""

from __future__ import annotations

import json
import time
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlencode

import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.resource import ResourceEditor
from ocp_resources.service import Service
from timeout_sampler import TimeoutSampler

from tests.ai_gateway.models_as_a_service.multitenancy.aitenant.utils import AIGATEWAY_BOOTSTRAPPED_TENANT_NAME
from tests.ai_gateway.models_as_a_service.observability.constants import (
    LOKI_GATEWAY_SERVICE_NAME,
    LOKI_QUERY_TENANT,
    LOKI_QUERY_URL,
    USAGE_LOGS_MULTITENANT_QUERY_LOOKBACK,
    USAGE_LOGS_PROPAGATION_SLEEP,
    USAGE_LOGS_SERVICE_NAME,
    USAGE_LOGS_TENANCY_PROXY_DEPLOYMENT,
    USAGE_LOGS_TENANCY_PROXY_LOCAL_PORT,
    USAGE_LOGS_TENANCY_PROXY_PORT,
)
from ocp_resources.exceptions import MissingResourceError
from tests.ai_gateway.models_as_a_service.utils import endpoints_have_ready_addresses
from tests.observability.authorization import authenticate_persona_tokens
from utilities.infra import get_dsci_monitoring_namespace
from utilities.resources.maastenantconfig import MaasTenantConfig

LOGGER = structlog.get_logger(name=__name__)

LOKI_QUERY_RANGE_URL: str = LOKI_QUERY_URL.replace("/loki/api/v1/query", "/loki/api/v1/query_range")
TENANCY_PROXY_QUERY_RANGE_URL: str = (
    f"https://localhost:{USAGE_LOGS_TENANCY_PROXY_LOCAL_PORT}/loki/api/v1/query_range"
)


def usage_logs_telemetry_spec(*, capture_user: bool = True) -> dict[str, Any]:
    """Return MaasTenantConfig telemetry spec for usage-log E2E."""
    return {
        "enabled": True,
        "logs": {
            "captureUser": capture_user,
        },
    }


@contextmanager
def patch_maastenantconfig_usage_telemetry(
    maas_tenant_config: MaasTenantConfig,
    *,
    capture_user: bool = True,
) -> Generator[MaasTenantConfig]:
    """Enable tenant telemetry and usage-log user capture for the test."""
    with ResourceEditor(
        patches={
            maas_tenant_config: {
                "spec": {
                    "telemetry": usage_logs_telemetry_spec(capture_user=capture_user),
                },
            },
        },
    ):
        yield maas_tenant_config


def usage_logs_tenancy_proxy_ready(
    admin_client: DynamicClient,
    monitoring_namespace: str,
) -> bool:
    """Return True when the usage-logs tenancy proxy Deployment is available."""
    deployment = Deployment(
        client=admin_client,
        name=USAGE_LOGS_TENANCY_PROXY_DEPLOYMENT,
        namespace=monitoring_namespace,
    )
    if not deployment.exists:
        return False
    deployment.get()
    status = deployment.instance.status
    return bool(status and status.availableReplicas and status.availableReplicas >= 1)


def loki_gateway_service_ready(
    admin_client: DynamicClient,
    monitoring_namespace: str,
) -> bool:
    """Return True when the Loki gateway Service exists and has ready endpoints."""
    service = Service(
        client=admin_client,
        name=LOKI_GATEWAY_SERVICE_NAME,
        namespace=monitoring_namespace,
    )
    if not service.exists:
        return False
    return endpoints_have_ready_addresses(
        admin_client=admin_client,
        namespace=monitoring_namespace,
        name=LOKI_GATEWAY_SERVICE_NAME,
    )


def usage_logs_e2e_stack_ready(admin_client: DynamicClient) -> tuple[bool, str]:
    """Return whether Loki + tenancy proxy prerequisites are present."""
    try:
        monitoring_namespace = get_dsci_monitoring_namespace(client=admin_client)
    except MissingResourceError:
        return False, "DSCInitialization not found (cannot resolve monitoring namespace)"
    if not usage_logs_tenancy_proxy_ready(
        admin_client=admin_client,
        monitoring_namespace=monitoring_namespace,
    ):
        return (
            False,
            f"Deployment '{USAGE_LOGS_TENANCY_PROXY_DEPLOYMENT}' is not ready in '{monitoring_namespace}'",
        )
    if not loki_gateway_service_ready(
        admin_client=admin_client,
        monitoring_namespace=monitoring_namespace,
    ):
        return (
            False,
            f"Service '{LOKI_GATEWAY_SERVICE_NAME}' is not ready in '{monitoring_namespace}' (LokiStack not ready)",
        )
    return True, monitoring_namespace


def resolve_openshift_username(admin_client: DynamicClient, bearer_token: str) -> str:
    """Return the Kubernetes username for a bearer token via TokenReview."""
    identities = authenticate_persona_tokens(
        admin_client=admin_client,
        tokens={"caller": bearer_token},
    )
    return identities["caller"].principal


def subscription_count_logql(*, user_id: str, lookback: str = USAGE_LOGS_MULTITENANT_QUERY_LOOKBACK) -> str:
    """Build LogQL that counts usage-log lines per subscription for one user."""
    escaped_user = user_id.replace("\\", "\\\\").replace('"', '\\"')
    return (
        f'count by (subscription) (count_over_time('
        f'{{service_name="{USAGE_LOGS_SERVICE_NAME}"}} '
        f'| user_id="{escaped_user}" '
        f'| subscription!="" | subscription!="-" '
        f"[{lookback}]))"
    )


def _query_range_params(*, logql: str, lookback: str) -> dict[str, str]:
    end_ns = time.time_ns()
    start_ns = end_ns - _lookback_to_nanoseconds(lookback)
    return {
        "query": logql,
        "start": str(start_ns),
        "end": str(end_ns),
        "limit": "5000",
    }


def _lookback_to_nanoseconds(lookback: str) -> int:
    """Convert a simple LogQL-style lookback (e.g. 30m, 1h) to nanoseconds."""
    unit = lookback[-1]
    value = int(lookback[:-1])
    multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if unit not in multipliers:
        raise ValueError(f"unsupported lookback duration: {lookback}")
    return value * multipliers[unit] * 1_000_000_000


def query_loki_gateway_range(
    *,
    http_session: requests.Session,
    admin_token: str,
    logql: str,
    lookback: str = USAGE_LOGS_MULTITENANT_QUERY_LOOKBACK,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    """Run a LogQL query_range against the admin Loki gateway (requires Loki port-forward)."""
    params = _query_range_params(logql=logql, lookback=lookback)
    response = http_session.get(
        url=f"{LOKI_QUERY_RANGE_URL}?{urlencode(params)}",
        headers={
            "Authorization": f"Bearer {admin_token}",
            "X-Scope-OrgID": LOKI_QUERY_TENANT,
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "success":
        raise AssertionError(f"Loki query failed: {json.dumps(payload)[:500]}")
    return payload


def query_usage_logs_tenancy_proxy_range(
    *,
    http_session: requests.Session,
    logql: str,
    user_bearer_token: str,
    lookback: str = USAGE_LOGS_MULTITENANT_QUERY_LOOKBACK,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    """Run a LogQL query_range through the usage-logs tenancy proxy (requires proxy port-forward)."""
    params = _query_range_params(logql=logql, lookback=lookback)
    response = http_session.get(
        url=f"{TENANCY_PROXY_QUERY_RANGE_URL}?{urlencode(params)}",
        headers={"Authorization": f"Bearer {user_bearer_token}"},
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "success":
        raise AssertionError(f"Tenancy proxy query failed: {json.dumps(payload)[:500]}")
    return payload


def subscription_names_from_loki_response(payload: Mapping[str, Any]) -> set[str]:
    """Extract subscription label values from a Loki matrix/vector response."""
    data = payload.get("data") or {}
    result = data.get("result") or []
    names: set[str] = set()
    for series in result:
        metric = series.get("metric") or {}
        subscription = metric.get("subscription")
        if subscription and subscription not in {"", "-"}:
            names.add(str(subscription))
    return names


def wait_for_subscriptions_in_loki(
    *,
    http_session: requests.Session,
    admin_token: str,
    user_id: str,
    expected_subscriptions: set[str],
    query_via_proxy: bool,
    user_bearer_token: str,
    timeout: int,
    lookback: str = USAGE_LOGS_MULTITENANT_QUERY_LOOKBACK,
) -> set[str]:
    """Poll until Loki returns all expected subscription names for the user."""
    logql = subscription_count_logql(user_id=user_id, lookback=lookback)
    subscriptions_from_last_poll: set[str] = set()

    def subscriptions_visible() -> bool:
        nonlocal subscriptions_from_last_poll
        if query_via_proxy:
            payload = query_usage_logs_tenancy_proxy_range(
                http_session=http_session,
                logql=logql,
                user_bearer_token=user_bearer_token,
                lookback=lookback,
            )
        else:
            payload = query_loki_gateway_range(
                http_session=http_session,
                admin_token=admin_token,
                logql=logql,
                lookback=lookback,
            )
        subscriptions_from_last_poll = subscription_names_from_loki_response(payload=payload)
        return expected_subscriptions.issubset(subscriptions_from_last_poll)

    for ready in TimeoutSampler(
        wait_timeout=timeout,
        sleep=USAGE_LOGS_PROPAGATION_SLEEP,
        func=subscriptions_visible,
    ):
        if ready:
            return subscriptions_from_last_poll
    raise AssertionError(
        f"Timed out waiting for subscriptions {sorted(expected_subscriptions)} "
        f"for user_id={user_id!r} (via_proxy={query_via_proxy})"
    )


def maastenantconfig_for_namespace(
    admin_client: DynamicClient,
    tenant_namespace: str,
) -> MaasTenantConfig:
    """Return MaasTenantConfig/default-tenant in a tenant namespace."""
    config = MaasTenantConfig(
        client=admin_client,
        name=AIGATEWAY_BOOTSTRAPPED_TENANT_NAME,
        namespace=tenant_namespace,
    )
    assert config.exists, (
        f"MaasTenantConfig/{AIGATEWAY_BOOTSTRAPPED_TENANT_NAME} not found in '{tenant_namespace}'"
    )
    return config
