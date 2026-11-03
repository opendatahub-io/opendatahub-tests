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
from ocp_resources.custom_resource_definition import CustomResourceDefinition
from ocp_resources.deployment import Deployment
from ocp_resources.dsc_initialization import DSCInitialization
from ocp_resources.exceptions import MissingResourceError
from ocp_resources.namespace import Namespace
from ocp_resources.resource import ResourceEditor
from ocp_resources.service import Service
from pytest_testconfig import config as py_config
from timeout_sampler import TimeoutSampler

from tests.ai_gateway.models_as_a_service.multitenancy.aitenant.utils import AIGATEWAY_BOOTSTRAPPED_TENANT_NAME
from tests.ai_gateway.models_as_a_service.observability.constants import (
    LOKI_GATEWAY_SERVICE_NAME,
    LOKI_QUERY_TENANT,
    LOKI_QUERY_URL,
    LOKI_STACK_CRD_NAME,
    MONITORING_CRD_NAME,
    OTEL_COLLECTOR_CRD_NAME,
    USAGE_LOGS_MULTITENANT_QUERY_LOOKBACK,
    USAGE_LOGS_PROPAGATION_SLEEP,
    USAGE_LOGS_SERVICE_NAME,
    USAGE_LOGS_TENANCY_PROXY_DEPLOYMENT,
    USAGE_LOGS_TENANCY_PROXY_LOCAL_PORT,
)
from tests.ai_gateway.models_as_a_service.observability.utils import resolve_maas_monitoring_namespace
from tests.ai_gateway.models_as_a_service.utils import endpoints_have_ready_addresses
from tests.observability.authorization import authenticate_persona_tokens
from utilities.infra import get_dsci_monitoring_namespace
from utilities.resources.maastenantconfig import MaasTenantConfig

LOGGER = structlog.get_logger(name=__name__)

LOKI_QUERY_RANGE_URL: str = LOKI_QUERY_URL.replace("/loki/api/v1/query", "/loki/api/v1/query_range")
TENANCY_PROXY_QUERY_RANGE_URL: str = f"https://localhost:{USAGE_LOGS_TENANCY_PROXY_LOCAL_PORT}/loki/api/v1/query_range"


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


_DSCI_USAGE_LOGS_PIPELINE_CONDITIONS: tuple[str, ...] = (
    "MonitoringReady",
    "MonitoringStackAvailable",
    "LokiStackAvailable",
    "UsageLogsCollectorAvailable",
)

_USAGE_LOGS_DSCI_FIX_HINT = (
    "Fix: on DSCInitialization spec.monitoring set managementState=Managed and configure "
    "usageLogs.storage (S3 secret in the monitoring namespace) so the ODH operator deploys "
    "data-science-lokistack and the usage-logs collector. Pipeline clusters must provide this "
    "before running multitenancy usage-log E2E (RHOAIENG-99404)."
)


def _dsci_condition_fields(
    dsci: DSCInitialization,
    condition_type: str,
) -> tuple[str | None, str | None, str | None]:
    """Return (status, reason, message) for a DSCI status condition type."""
    for condition in dsci.instance.status.conditions or []:
        if condition.type == condition_type:
            return (
                str(condition.status) if condition.status is not None else None,
                str(condition.reason) if condition.reason is not None else None,
                str(condition.message) if condition.message is not None else None,
            )
    return None, None, None


def verify_usage_logs_multi_tenant_e2e_prerequisites(
    admin_client: DynamicClient,
) -> tuple[str | None, str | None]:
    """Verify pipeline prerequisites for multitenancy usage-log E2E.

    Returns (monitoring_namespace, None) when ready, else (None, pytest.fail message).
    """
    dsci_name = py_config["dsci_name"]
    dsci = DSCInitialization(client=admin_client, name=dsci_name)
    if not dsci.exists:
        return (
            None,
            f"DSCInitialization '{dsci_name}' not found. "
            f"Set pytest --tc=dsci_name=<name> to match the cluster (expected default-dsci on RHOAI).",
        )

    monitoring_spec = dsci.instance.spec.get("monitoring") or {}
    management_state = monitoring_spec.get("managementState")
    if management_state != "Managed":
        return (
            None,
            f"DSCInitialization '{dsci_name}' spec.monitoring.managementState is {management_state!r}, "
            f"expected 'Managed'. {_USAGE_LOGS_DSCI_FIX_HINT}",
        )

    for crd_name in (LOKI_STACK_CRD_NAME, OTEL_COLLECTOR_CRD_NAME, MONITORING_CRD_NAME):
        if not CustomResourceDefinition(client=admin_client, name=crd_name).exists:
            return (
                None,
                f"CRD '{crd_name}' is not installed. "
                f"The usage-logs LokiStack requires the Loki operator and ODH monitoring CRDs. "
                f"{_USAGE_LOGS_DSCI_FIX_HINT}",
            )

    for condition_type in _DSCI_USAGE_LOGS_PIPELINE_CONDITIONS:
        status, reason, message = _dsci_condition_fields(dsci=dsci, condition_type=condition_type)
        if status != "True":
            detail = f"type={condition_type} status={status!r} reason={reason!r} message={message!r}"
            return (
                None,
                f"DSCInitialization '{dsci_name}' is not ready for usage-log E2E ({detail}). "
                f"{_USAGE_LOGS_DSCI_FIX_HINT}",
            )

    try:
        monitoring_namespace = get_dsci_monitoring_namespace(client=admin_client)
    except MissingResourceError:
        return (
            None,
            f"Could not resolve monitoring namespace from DSCInitialization '{dsci_name}'. "
            f"Ensure spec.monitoring.namespace is set (typically redhat-ods-monitoring).",
        )

    if not Namespace(client=admin_client, name=monitoring_namespace).exists:
        return (
            None,
            f"Monitoring namespace '{monitoring_namespace}' does not exist. "
            f"DSCI bootstrap should create it when monitoring is Managed.",
        )

    maas_monitoring_namespace = resolve_maas_monitoring_namespace(admin_client=admin_client)
    if maas_monitoring_namespace and maas_monitoring_namespace != monitoring_namespace:
        return (
            None,
            f"maas-controller MONITORING_NAMESPACE ({maas_monitoring_namespace!r}) does not match "
            f"DSCI monitoring namespace ({monitoring_namespace!r}). Align operator env and DSCI.",
        )

    if not loki_gateway_service_ready(
        admin_client=admin_client,
        monitoring_namespace=monitoring_namespace,
    ):
        return (
            None,
            f"Service '{LOKI_GATEWAY_SERVICE_NAME}' has no ready endpoints in '{monitoring_namespace}'. "
            f"Wait for data-science-lokistack gateway after LokiStackAvailable=True, or fix LokiStack reconcile. "
            f"{_USAGE_LOGS_DSCI_FIX_HINT}",
        )

    if not usage_logs_tenancy_proxy_ready(
        admin_client=admin_client,
        monitoring_namespace=monitoring_namespace,
    ):
        return (
            None,
            f"Deployment '{USAGE_LOGS_TENANCY_PROXY_DEPLOYMENT}' is not Available in '{monitoring_namespace}'. "
            f"Ensure maas-controller reconciled usage-logs-tenancy-proxy (Config/default present).",
        )

    return monitoring_namespace, None


def usage_logs_e2e_stack_ready(admin_client: DynamicClient) -> tuple[bool, str]:
    """Return whether Loki + tenancy proxy prerequisites are present (legacy helper)."""
    monitoring_namespace, failure = verify_usage_logs_multi_tenant_e2e_prerequisites(admin_client=admin_client)
    if failure:
        return False, failure
    assert monitoring_namespace is not None
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
        f"count by (subscription) (count_over_time("
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
    assert config.exists, f"MaasTenantConfig/{AIGATEWAY_BOOTSTRAPPED_TENANT_NAME} not found in '{tenant_namespace}'"
    return config
