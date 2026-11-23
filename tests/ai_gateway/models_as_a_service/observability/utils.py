"""Utilities for MaaS observability integration tests."""

from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

import portforward
import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.cluster_role_binding import ClusterRoleBinding
from ocp_resources.custom_resource_definition import CustomResourceDefinition
from ocp_resources.deployment import Deployment
from ocp_resources.dsc_initialization import DSCInitialization
from ocp_resources.namespace import Namespace
from ocp_resources.open_telemetry_collector import OpenTelemetryCollector
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from ocp_resources.service_monitor import ServiceMonitor
from pytest_testconfig import config as py_config
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_gateway.models_as_a_service.observability.constants import (
    DEFAULT_LIMITADOR_SCRAPE_INTERVAL,
    DSCI_USAGE_LOGS_CONDITIONS,
    DSCI_USAGE_LOGS_CONDITIONS_WAIT_TIMEOUT,
    LIMITADOR_APP_LABEL,
    LIMITADOR_DEPLOYMENT_NAMESPACES,
    LIMITADOR_METRICS_PATH,
    LIMITADOR_METRICS_PORT,
    LIMITADOR_SCRAPE_LABEL,
    LIMITADOR_SCRAPE_LABEL_VALUE,
    LIMITADOR_SERVICE_MONITOR_NAME,
    LIMITADOR_SERVICE_MONITOR_WAIT_TIMEOUT,
    LOKI_GATEWAY_SERVICE_NAME,
    LOKI_GATEWAY_SERVICE_PORT,
    LOKI_QUERY_LOCAL_PORT,
    LOKI_QUERY_TENANT,
    LOKI_QUERY_URL,
    OTEL_COLLECTOR_CRD_NAME,
    USAGE_LOGGING_RESOURCES_WAIT_TIMEOUT,
    USAGE_LOGS_COLLECTOR_NAME,
    USAGE_LOGS_CRB_NAME,
    USAGE_LOGS_ENVOY_FILTER_NAME,
    USAGE_LOGS_PROPAGATION_SLEEP,
    USAGE_LOGS_PROPAGATION_TIMEOUT,
    loki_usage_rate_limited_requests_query,
    loki_usage_requests_query,
    loki_usage_successful_requests_query,
    loki_usage_tokens_query,
)
from tests.ai_gateway.models_as_a_service.utils import (
    get_total_tokens,
    verify_chat_completions,
)
from utilities.constants import MAAS_GATEWAY_NAMESPACE
from utilities.resources.envoy_filter import EnvoyFilter
from utilities.resources.kuadrant import Kuadrant
from utilities.resources.maas_config import Config as MaaSConfig

LOGGER = structlog.get_logger(name=__name__)

MAAS_CONTROLLER_DEPLOYMENT_NAME = "maas-controller"


@contextmanager
def maas_config_usage_logging_enabled(
    maas_config: MaaSConfig,
) -> Generator[MaaSConfig]:
    """Patch Config/default to enable usageLogging and restore on exit."""
    with ResourceEditor(patches={maas_config: {"spec": {"usageLogging": True}}}):
        yield maas_config


@contextmanager
def maas_config_usage_logging_disabled(
    maas_config: MaaSConfig,
) -> Generator[MaaSConfig]:
    """Patch Config/default to disable usageLogging; no-op when already disabled."""
    if not usage_logging_enabled(maas_config=maas_config):
        yield maas_config
        return

    with ResourceEditor(patches={maas_config: {"spec": {"usageLogging": False}}}):
        yield maas_config


@contextmanager
def patch_maas_config_limitador_scrape_interval(
    maas_config: MaaSConfig,
    scrape_interval: str,
) -> Generator[MaaSConfig]:
    """Patch Config/default limitadorScrapeInterval and restore the prior spec on exit."""
    with ResourceEditor(patches={maas_config: {"spec": {"limitadorScrapeInterval": scrape_interval}}}):
        yield maas_config


def get_maas_controller_env_var(admin_client: DynamicClient, env_name: str) -> str:
    """Return an environment variable value from the maas-controller Deployment."""
    applications_namespace = py_config["applications_namespace"]
    controller_deployment = Deployment(
        client=admin_client,
        name=MAAS_CONTROLLER_DEPLOYMENT_NAME,
        namespace=applications_namespace,
        ensure_exists=True,
    )
    for container in controller_deployment.instance.spec.template.spec.containers:
        for env_var in container.env or []:
            if env_var.name == env_name:
                return (env_var.value or "").strip()
    return ""


def resolve_maas_monitoring_namespace(admin_client: DynamicClient) -> str:
    """Resolve the monitoring namespace configured on maas-controller."""
    monitoring_namespace = get_maas_controller_env_var(
        admin_client=admin_client,
        env_name="MONITORING_NAMESPACE",
    )
    if monitoring_namespace:
        return monitoring_namespace

    applications_namespace = py_config["applications_namespace"]
    if applications_namespace == "redhat-ods-applications":
        return "redhat-ods-monitoring"
    if applications_namespace == "opendatahub":
        return "redhat-ods-monitoring"
    return ""


def monitoring_namespace_exists(admin_client: DynamicClient, namespace_name: str) -> bool:
    """Return True when the monitoring namespace exists on the cluster."""
    return Namespace(client=admin_client, name=namespace_name).exists


def limitador_is_deployed(admin_client: DynamicClient) -> bool:
    """Return True when a Limitador pod is running in a known policy-engine namespace."""
    for namespace_name in LIMITADOR_DEPLOYMENT_NAMESPACES:
        pods = list(
            Pod.get(
                client=admin_client,
                namespace=namespace_name,
                label_selector=f"app={LIMITADOR_APP_LABEL}",
            )
        )
        if pods:
            return True
    return False


def resolve_kuadrant_namespace(admin_client: DynamicClient) -> str:
    """Return the namespace of the Kuadrant CR that hosts the Limitador policy engine."""
    kuadrant_instances = list(Kuadrant.get(client=admin_client))
    if not kuadrant_instances:
        raise AssertionError(
            "Kuadrant CR not found cluster-wide; cannot validate Limitador ServiceMonitor namespaceSelector"
        )

    preferred_namespaces = set(LIMITADOR_DEPLOYMENT_NAMESPACES)
    for kuadrant in kuadrant_instances:
        if kuadrant.namespace in preferred_namespaces:
            return kuadrant.namespace

    return kuadrant_instances[0].namespace


def limitador_namespace_selector_matches_kuadrant_namespace(
    namespace_selector: object,
    kuadrant_namespace: str,
) -> bool:
    """Return True when the ServiceMonitor namespaceSelector includes the Kuadrant namespace."""
    if getattr(namespace_selector, "any", False):
        return True

    match_names = getattr(namespace_selector, "matchNames", None) or []
    return kuadrant_namespace in match_names


def get_maas_config_default(admin_client: DynamicClient) -> MaaSConfig:
    """Return the cluster-scoped MaaS Config/default anchor."""
    maas_config = MaaSConfig(
        client=admin_client,
        name="default",
        ensure_exists=True,
    )
    assert maas_config.exists, "MaaS Config/default not found — maas-controller observability reconcile did not run"
    return maas_config


def expected_limitador_scrape_interval(maas_config: MaaSConfig) -> str:
    """Return the Limitador scrape interval from Config/default or the controller default."""
    spec = maas_config.instance.spec
    interval = getattr(spec, "limitadorScrapeInterval", None)
    if not interval:
        return DEFAULT_LIMITADOR_SCRAPE_INTERVAL
    return str(interval)


def usage_logging_enabled(maas_config: MaaSConfig) -> bool:
    """Return True when Config/default has usageLogging enabled."""
    spec = maas_config.instance.spec
    return bool(getattr(spec, "usageLogging", False))


def config_owner_reference_present(
    owner_references: list[object] | None,
    config_uid: str,
) -> bool:
    """Return True when ownerReferences include the MaaS Config/default UID."""
    if not owner_references:
        return False
    return any(owner_ref.uid == config_uid for owner_ref in owner_references)


def config_controller_owner_reference_present(
    owner_references: list[object] | None,
    config_uid: str,
) -> bool:
    """Return True when ownerReferences include Config/default as controller owner."""
    if not owner_references:
        return False
    return any(owner_ref.uid == config_uid and owner_ref.controller is True for owner_ref in owner_references)


def opentelemetry_collector_crd_installed(admin_client: DynamicClient) -> bool:
    """Return True when the OpenTelemetryCollector CRD is registered on the cluster."""
    otel_crd = CustomResourceDefinition(client=admin_client, name=OTEL_COLLECTOR_CRD_NAME)
    return bool(otel_crd.exists)


def usage_logs_collector_exists(
    admin_client: DynamicClient,
    namespace: str,
    name: str = USAGE_LOGS_COLLECTOR_NAME,
) -> bool:
    """Return True when a usage-logs OpenTelemetryCollector CR exists."""
    if not opentelemetry_collector_crd_installed(admin_client=admin_client):
        return False
    collector = OpenTelemetryCollector(client=admin_client, name=name, namespace=namespace)
    return bool(collector.exists)


def wait_for_limitador_service_monitor(
    admin_client: DynamicClient,
    monitoring_namespace: str,
    timeout: int = LIMITADOR_SERVICE_MONITOR_WAIT_TIMEOUT,
) -> ServiceMonitor | None:
    """Wait for maas-controller to create the Limitador ServiceMonitor."""
    service_monitor = ServiceMonitor(
        client=admin_client,
        name=LIMITADOR_SERVICE_MONITOR_NAME,
        namespace=monitoring_namespace,
    )
    for sampler in TimeoutSampler(
        wait_timeout=timeout,
        sleep=5,
        func=lambda: service_monitor.exists,
    ):
        if sampler:
            return service_monitor
    return None


def validate_limitador_service_monitor_spec(
    admin_client: DynamicClient,
    limitador_service_monitor: ServiceMonitor,
    maas_config: MaaSConfig,
) -> None:
    """Verify the Limitador ServiceMonitor matches ensureLimitadorServiceMonitor expectations."""
    assert limitador_service_monitor.exists, (
        f"ServiceMonitor '{limitador_service_monitor.name}' not found in '{limitador_service_monitor.namespace}'"
    )

    metadata = limitador_service_monitor.instance.metadata
    labels = dict(metadata.labels or {})
    assert labels.get("app") == LIMITADOR_APP_LABEL
    assert labels.get(LIMITADOR_SCRAPE_LABEL) == LIMITADOR_SCRAPE_LABEL_VALUE

    config_uid = maas_config.instance.metadata.uid
    assert config_owner_reference_present(
        owner_references=metadata.ownerReferences,
        config_uid=config_uid,
    ), f"Expected Config/default UID '{config_uid}' in ServiceMonitor ownerReferences"

    spec = limitador_service_monitor.instance.spec
    assert len(spec.endpoints) == 1, f"Expected 1 scrape endpoint, found {len(spec.endpoints)}"

    endpoint = spec.endpoints[0]
    assert endpoint.path == LIMITADOR_METRICS_PATH
    assert endpoint.port == LIMITADOR_METRICS_PORT
    assert str(endpoint.interval) == expected_limitador_scrape_interval(maas_config=maas_config)

    selector_labels = dict(spec.selector.matchLabels or {})
    assert selector_labels.get("app") == LIMITADOR_APP_LABEL

    kuadrant_namespace = resolve_kuadrant_namespace(admin_client=admin_client)
    namespace_selector = spec.namespaceSelector
    assert limitador_namespace_selector_matches_kuadrant_namespace(
        namespace_selector=namespace_selector,
        kuadrant_namespace=kuadrant_namespace,
    ), (
        f"Expected ServiceMonitor namespaceSelector to match Kuadrant namespace "
        f"'{kuadrant_namespace}', got namespaceSelector={namespace_selector}"
    )


def usage_logging_resources_in_expected_state(
    admin_client: DynamicClient,
    monitoring_namespace: str,
    present: bool,
) -> bool:
    """Return True when usage-log resources all exist or are all absent as expected."""
    usage_logs_envoy_filter = EnvoyFilter(
        client=admin_client,
        name=USAGE_LOGS_ENVOY_FILTER_NAME,
        namespace=MAAS_GATEWAY_NAMESPACE,
    )
    usage_logs_crb = ClusterRoleBinding(
        client=admin_client,
        name=USAGE_LOGS_CRB_NAME,
    )
    all_present = (
        usage_logs_envoy_filter.exists
        and usage_logs_collector_exists(
            admin_client=admin_client,
            namespace=monitoring_namespace,
        )
        and usage_logs_crb.exists
    )
    return all_present if present else not all_present


def wait_for_usage_logging_resources(
    admin_client: DynamicClient,
    monitoring_namespace: str,
    timeout: int = USAGE_LOGGING_RESOURCES_WAIT_TIMEOUT,
) -> None:
    """Wait for maas-controller to deploy usage-log observability resources."""
    for resources_ready in TimeoutSampler(
        wait_timeout=timeout,
        sleep=5,
        func=lambda: usage_logging_resources_in_expected_state(
            admin_client=admin_client,
            monitoring_namespace=monitoring_namespace,
            present=True,
        ),
    ):
        if resources_ready:
            return


def wait_for_usage_logging_resources_absent(
    admin_client: DynamicClient,
    monitoring_namespace: str,
    timeout: int = USAGE_LOGGING_RESOURCES_WAIT_TIMEOUT,
) -> None:
    """Wait for maas-controller to remove usage-log observability resources."""
    for resources_removed in TimeoutSampler(
        wait_timeout=timeout,
        sleep=5,
        func=lambda: usage_logging_resources_in_expected_state(
            admin_client=admin_client,
            monitoring_namespace=monitoring_namespace,
            present=False,
        ),
    ):
        if resources_removed:
            return


def validate_usage_logging_resources(
    admin_client: DynamicClient,
    maas_config: MaaSConfig,
    monitoring_namespace: str,
) -> None:
    """Validate usage-log observability resource ownerReferences after resources are present."""
    assert usage_logging_enabled(maas_config=maas_config), "Expected Config/default.spec.usageLogging to be enabled"

    config_uid = maas_config.instance.metadata.uid

    usage_logs_envoy_filter = EnvoyFilter(
        client=admin_client,
        name=USAGE_LOGS_ENVOY_FILTER_NAME,
        namespace=MAAS_GATEWAY_NAMESPACE,
    )
    assert config_owner_reference_present(
        owner_references=usage_logs_envoy_filter.instance.metadata.ownerReferences,
        config_uid=config_uid,
    ), f"Expected Config/default UID '{config_uid}' in EnvoyFilter '{USAGE_LOGS_ENVOY_FILTER_NAME}' ownerReferences"

    if not opentelemetry_collector_crd_installed(admin_client=admin_client):
        raise AssertionError("OpenTelemetryCollector CRD not installed; usage-logs collector cannot be validated")

    collector = OpenTelemetryCollector(
        client=admin_client,
        name=USAGE_LOGS_COLLECTOR_NAME,
        namespace=monitoring_namespace,
    )
    assert config_controller_owner_reference_present(
        owner_references=collector.instance.metadata.ownerReferences,
        config_uid=config_uid,
    ), (
        f"Expected Config/default UID '{config_uid}' in OpenTelemetryCollector "
        f"'{USAGE_LOGS_COLLECTOR_NAME}' ownerReferences"
    )

    usage_logs_crb = ClusterRoleBinding(
        client=admin_client,
        name=USAGE_LOGS_CRB_NAME,
    )
    assert config_controller_owner_reference_present(
        owner_references=usage_logs_crb.instance.metadata.ownerReferences,
        config_uid=config_uid,
    ), f"Expected Config/default UID '{config_uid}' in ClusterRoleBinding '{USAGE_LOGS_CRB_NAME}' ownerReferences"


def verify_usage_logging_resources_deployed(
    admin_client: DynamicClient,
    maas_config: MaaSConfig,
    monitoring_namespace: str,
) -> None:
    """Wait for and validate the usage-log observability stack after usageLogging is enabled."""
    wait_for_usage_logging_resources(
        admin_client=admin_client,
        monitoring_namespace=monitoring_namespace,
    )
    validate_usage_logging_resources(
        admin_client=admin_client,
        maas_config=maas_config,
        monitoring_namespace=monitoring_namespace,
    )


def verify_usage_logging_resources_removed(
    admin_client: DynamicClient,
    maas_config: MaaSConfig,
    monitoring_namespace: str,
) -> None:
    """Wait for usage-log observability resources to be removed after usageLogging is disabled."""
    assert not usage_logging_enabled(maas_config=maas_config), (
        "Expected Config/default.spec.usageLogging to be disabled"
    )
    wait_for_usage_logging_resources_absent(
        admin_client=admin_client,
        monitoring_namespace=monitoring_namespace,
    )


def wait_for_limitador_scrape_interval(
    admin_client: DynamicClient,
    monitoring_namespace: str,
    expected_interval: str,
    timeout: int = LIMITADOR_SERVICE_MONITOR_WAIT_TIMEOUT,
) -> ServiceMonitor:
    """Wait for the Limitador ServiceMonitor scrape interval to match Config/default."""
    service_monitor = ServiceMonitor(
        client=admin_client,
        name=LIMITADOR_SERVICE_MONITOR_NAME,
        namespace=monitoring_namespace,
    )

    def scrape_interval_matches() -> bool:
        if not service_monitor.exists:
            return False
        service_monitor.get()
        endpoint = service_monitor.instance.spec.endpoints[0]
        return str(endpoint.interval) == expected_interval

    for interval_matches in TimeoutSampler(
        wait_timeout=timeout,
        sleep=5,
        func=scrape_interval_matches,
    ):
        if interval_matches:
            return service_monitor
    return service_monitor


def verify_limitador_scrape_interval_on_servicemonitor(
    admin_client: DynamicClient,
    monitoring_namespace: str,
    expected_interval: str,
) -> None:
    """Wait for the Limitador ServiceMonitor scrape interval to match Config/default."""
    wait_for_limitador_scrape_interval(
        admin_client=admin_client,
        monitoring_namespace=monitoring_namespace,
        expected_interval=expected_interval,
    )


def assert_usage_logging_resources_absent(
    admin_client: DynamicClient,
    maas_config: MaaSConfig,
    monitoring_namespace: str,
) -> None:
    """Verify usage-log observability resources are absent when usageLogging is disabled."""
    assert not usage_logging_enabled(maas_config=maas_config), (
        "Expected Config/default.spec.usageLogging to be disabled"
    )

    assert usage_logging_resources_in_expected_state(
        admin_client=admin_client,
        monitoring_namespace=monitoring_namespace,
        present=False,
    ), "Usage-log observability resources must be absent when usageLogging is disabled"


def usage_logs_monitoring_spec(secret_name: str, storage_class_name: str) -> dict[str, Any]:
    """Build the DSCI spec.monitoring patch that enables usage logs on S3-backed storage."""
    return {
        "managementState": "Managed",
        "usageLogs": {
            "storage": {
                "type": "s3",
                "secretName": secret_name,
                "credentialMode": "static",
                "storageClassName": storage_class_name,
            },
        },
    }


def wait_for_dsci_usage_logs_conditions(
    dsci_resource: DSCInitialization,
    since: datetime | None = None,
    timeout: int = DSCI_USAGE_LOGS_CONDITIONS_WAIT_TIMEOUT,
) -> None:
    """Wait for DSCI to report the odh-observability usage-logs conditions as True.

    opendatahub-operator projects the Monitoring CR's status onto DSCI.status.conditions, so
    these conditions are awaited on DSCI directly rather than by watching the Monitoring CR.

    Args:
        dsci_resource: DSCI resource whose conditions are awaited.
        since: When set, a condition only satisfies the wait if it transitioned to True at
            or after this instant. Pass the time spec.monitoring was patched so the
            conditions left behind by a previous monitoring configuration do not satisfy
            the wait before the operator has rolled out the newly configured
            usage-logs collector and LokiStack.
        timeout: Maximum number of seconds to wait for all conditions.
    """

    def usage_logs_conditions_ready() -> bool:
        dsci_status = dsci_resource.instance.get("status") or {}
        conditions = {condition["type"]: condition for condition in dsci_status.get("conditions") or []}
        for condition_name in DSCI_USAGE_LOGS_CONDITIONS:
            condition = conditions.get(condition_name)
            if not condition or condition.get("status") != "True":
                return False
            # Ready describes the whole DSCI and does not transition for a monitoring-only
            # change, so only the usage-logs conditions need to be fresh.
            if since is None or condition_name == "Ready":
                continue
            transition_time = condition.get("lastTransitionTime")
            if not transition_time or datetime.fromisoformat(transition_time) < since:
                return False
        return True

    try:
        for ready in TimeoutSampler(wait_timeout=timeout, sleep=5, func=usage_logs_conditions_ready):
            if ready:
                return
    except TimeoutExpiredError as err:
        raise AssertionError(
            f"DSCI conditions {DSCI_USAGE_LOGS_CONDITIONS} did not become True within {timeout}s"
        ) from err


@contextmanager
def loki_query_port_forward(namespace: str) -> Generator[None]:
    """Forward the local Loki query port to the usage-logs LokiStack gateway service."""
    with portforward.forward(
        pod_or_service=LOKI_GATEWAY_SERVICE_NAME,
        namespace=namespace,
        from_port=LOKI_QUERY_LOCAL_PORT,
        to_port=LOKI_GATEWAY_SERVICE_PORT,
        waiting=20,
    ):
        yield


def query_loki_instant(
    http_session: requests.Session,
    admin_token: str,
    logql_query: str,
) -> dict[str, Any]:
    """Run a Loki instant query against the usage-logs LokiStack application tenant."""
    response = http_session.get(
        url=LOKI_QUERY_URL,
        headers={
            "Authorization": f"Bearer {admin_token}",
            "X-Scope-OrgID": LOKI_QUERY_TENANT,
        },
        params={"query": logql_query},
        timeout=60,
    )
    assert response.status_code == 200, (
        f"Loki query failed: HTTP {response.status_code} {response.text[:200]} (query={logql_query})"
    )
    return response.json()


def loki_query_value(query_response: dict[str, Any]) -> int:
    """Return the numeric value of a Loki instant query; an empty result counts as 0."""
    if query_response.get("status") != "success":
        pytest.fail(f"Loki query did not succeed: {query_response}")
    results = query_response["data"]["result"]
    if not results:
        return 0
    return int(float(results[0]["value"][1]))


def get_loki_usage_totals(
    http_session: requests.Session,
    admin_token: str,
    model: str,
) -> tuple[int, int, int]:
    """Return the (requests, successful requests, total tokens) usage-logged for a model in the query window."""
    requests_total = loki_query_value(
        query_response=query_loki_instant(
            http_session=http_session,
            admin_token=admin_token,
            logql_query=loki_usage_requests_query(model=model),
        ),
    )
    successful_requests_total = loki_query_value(
        query_response=query_loki_instant(
            http_session=http_session,
            admin_token=admin_token,
            logql_query=loki_usage_successful_requests_query(model=model),
        ),
    )
    tokens_total = loki_query_value(
        query_response=query_loki_instant(
            http_session=http_session,
            admin_token=admin_token,
            logql_query=loki_usage_tokens_query(model=model),
        ),
    )
    LOGGER.info(
        f"Usage logs for model '{model}': requests={requests_total}, "
        f"successful_requests={successful_requests_total}, tokens={tokens_total}"
    )
    return requests_total, successful_requests_total, tokens_total


def get_loki_rate_limited_requests(
    http_session: requests.Session,
    admin_token: str,
    subscription: str,
) -> int:
    """Return the count of usage-logged rate-limited (429) inference requests for a subscription."""
    rate_limited_requests_total = loki_query_value(
        query_response=query_loki_instant(
            http_session=http_session,
            admin_token=admin_token,
            logql_query=loki_usage_rate_limited_requests_query(subscription=subscription),
        ),
    )
    LOGGER.info(f"Usage logs rate-limited requests for subscription '{subscription}': {rate_limited_requests_total}")
    return rate_limited_requests_total


def get_loki_usage_totals_with_rate_limits(
    http_session: requests.Session,
    admin_token: str,
    model: str,
    subscription: str,
) -> tuple[int, int, int]:
    """Return the (successful requests, rate-limited requests, total tokens) usage-logged for a model.

    Extends get_loki_usage_totals with the rate-limited request count. That count is not
    model-scoped: a rate-limited call is rejected before a model response exists, so its
    `model` label is not guaranteed to match the served model name; it is scoped by the
    test's subscription instead.
    """
    _, successful_requests_total, tokens_total = get_loki_usage_totals(
        http_session=http_session,
        admin_token=admin_token,
        model=model,
    )
    rate_limited_requests_total = get_loki_rate_limited_requests(
        http_session=http_session,
        admin_token=admin_token,
        subscription=subscription,
    )
    return successful_requests_total, rate_limited_requests_total, tokens_total


def get_loki_usage_totals_snapshot(
    http_session: requests.Session,
    admin_token: str,
    model: str,
    subscription: str | None = None,
) -> tuple[int, int, int]:
    """Snapshot the Loki usage totals a test compares against after sending inference traffic.

    Without a subscription the snapshot is (requests, successful requests, total tokens);
    with one it is (successful requests, rate-limited requests, total tokens), so the same
    helper serves both the plain and the rate-limited usage-log flows.

    Args:
        http_session: HTTP session used to query Loki.
        admin_token: Token used to authenticate the Loki queries.
        model: Served model name the usage totals are filtered by.
        subscription: When set, the snapshot also includes the rate-limited request count
            scoped to this subscription.

    Returns:
        The current usage totals in the shape described above.
    """
    if subscription is None:
        return get_loki_usage_totals(
            http_session=http_session,
            admin_token=admin_token,
            model=model,
        )
    return get_loki_usage_totals_with_rate_limits(
        http_session=http_session,
        admin_token=admin_token,
        model=model,
        subscription=subscription,
    )


def wait_for_loki_usage_totals(
    http_session: requests.Session,
    admin_token: str,
    model: str,
    expected_totals: tuple[int, int, int],
    subscription: str | None = None,
    timeout_context: str = "",
    timeout: int = USAGE_LOGS_PROPAGATION_TIMEOUT,
    sleep: int = USAGE_LOGS_PROPAGATION_SLEEP,
) -> None:
    """Wait until the Loki usage totals for a model reach the expected values.

    Usage records propagate asynchronously (gateway EnvoyFilter -> OTel collector -> Loki),
    so the totals are re-read until they match the expected values exactly.

    Args:
        http_session: HTTP session used to query Loki.
        admin_token: Token used to authenticate the Loki queries.
        model: Served model name the usage totals are filtered by.
        expected_totals: Totals the snapshot must reach, in the same shape
            get_loki_usage_totals_snapshot returns.
        subscription: When set, the snapshot is (successful requests, rate-limited requests,
            total tokens) scoped to this subscription; otherwise it is (requests, successful
            requests, total tokens).
        timeout_context: Optional phrase appended to the timeout message (e.g.
            "after streaming inference calls") to identify the flow that timed out.
        timeout: Maximum number of seconds to wait for the totals to match.
        sleep: Number of seconds between Loki queries.
    """
    if subscription is None:
        scope = f"model '{model}'"
        labels = ("requests", "successful_requests", "tokens")
    else:
        scope = f"model '{model}' and subscription '{subscription}'"
        labels = ("successful_requests", "rate_limited_requests", "tokens")

    observed: tuple[int | None, int | None, int | None] = (None, None, None)
    try:
        for observed in TimeoutSampler(
            wait_timeout=timeout,
            sleep=sleep,
            func=get_loki_usage_totals_snapshot,
            sensitive_keys=frozenset({"admin_token"}),
            http_session=http_session,
            admin_token=admin_token,
            model=model,
            subscription=subscription,
        ):
            if observed == expected_totals:
                return
    except TimeoutExpiredError:
        context_suffix = f" {timeout_context}" if timeout_context else ""
        expected_summary = ", ".join(f"{label}={value}" for label, value in zip(labels, expected_totals, strict=True))
        observed_summary = ", ".join(f"{label}={value}" for label, value in zip(labels, observed, strict=True))
        pytest.fail(
            f"Loki usage totals for {scope} did not reach the expected values within "
            f"{timeout}s{context_suffix}: "
            f"expected ({expected_summary}), "
            f"last observed ({observed_summary})"
        )


def send_non_streaming_inference_calls(
    request_session_http: requests.Session,
    model_url: str,
    headers: dict,
    model_id: str,
    logged_model: str,
    call_count: int,
    log_prefix: str,
    expected_status_codes: tuple[int, ...] = (200,),
) -> tuple[int, int, int]:
    """Send non-streaming chat-completion calls through the MaaS gateway and sum their usage.

    Every HTTP 200 response must report the served model name the Loki queries are filtered by,
    because the Loki `model` label is taken from the response body. HTTP 429 responses are
    counted as rate-limited calls and contribute neither tokens nor successful-request count.

    Args:
        request_session_http: HTTP session used to send the requests.
        model_url: Chat-completions URL of the model.
        headers: Request headers, including the MaaS authorization header.
        model_id: Model id sent in the request body.
        logged_model: Served model name the Loki queries are filtered by.
        call_count: Number of inference calls to send.
        log_prefix: Prefix used in log messages; the call index is appended to it.
        expected_status_codes: Status codes accepted for each call.

    Returns:
        The (successful calls, rate-limited calls, total tokens) observed across the calls.
    """
    successful_calls = 0
    rate_limited_calls = 0
    total_tokens = 0

    for call_index in range(call_count):
        response = verify_chat_completions(
            request_session_http=request_session_http,
            model_url=model_url,
            headers=headers,
            model_id=model_id,
            log_prefix=f"{log_prefix} [{call_index + 1}/{call_count}]",
            expected_status_codes=expected_status_codes,
        )
        if response.status_code == 429:
            rate_limited_calls += 1
            continue
        # The Loki `model` label is taken from the response body, so it must match the
        # served model name the queries are filtered by.
        response_model = response.json()["model"]
        assert response_model == logged_model, (
            f"Response model '{response_model}' does not match the served model name "
            f"'{logged_model}' the Loki queries are filtered by"
        )
        response_total_tokens = get_total_tokens(resp=response, fail_if_missing=True)
        assert response_total_tokens is not None
        total_tokens += response_total_tokens
        successful_calls += 1

    return successful_calls, rate_limited_calls, total_tokens
