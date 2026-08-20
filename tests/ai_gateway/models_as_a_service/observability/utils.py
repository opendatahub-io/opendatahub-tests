"""Utilities for MaaS observability integration tests."""

from collections.abc import Generator
from contextlib import contextmanager

import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.cluster_role_binding import ClusterRoleBinding
from ocp_resources.custom_resource_definition import CustomResourceDefinition
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.open_telemetry_collector import OpenTelemetryCollector
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from ocp_resources.service_monitor import ServiceMonitor
from ocp_utilities.monitoring import Prometheus
from pytest_testconfig import config as py_config
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_gateway.models_as_a_service.observability.constants import (
    DEFAULT_LIMITADOR_SCRAPE_INTERVAL,
    LIMITADOR_APP_LABEL,
    LIMITADOR_DEPLOYMENT_NAMESPACES,
    LIMITADOR_METRICS_PATH,
    LIMITADOR_METRICS_PORT,
    LIMITADOR_SCRAPE_LABEL,
    LIMITADOR_SCRAPE_LABEL_VALUE,
    LIMITADOR_SERVICE_MONITOR_NAME,
    LIMITADOR_SERVICE_MONITOR_WAIT_TIMEOUT,
    METRICS_POLL_TIMEOUT,
    OTEL_COLLECTOR_CRD_NAME,
    USAGE_LOGGING_RESOURCES_WAIT_TIMEOUT,
    USAGE_LOGS_COLLECTOR_NAME,
    USAGE_LOGS_CRB_NAME,
    USAGE_LOGS_ENVOY_FILTER_NAME,
    limitador_scrape_target_up_query,
)
from utilities.constants import MAAS_GATEWAY_NAMESPACE
from utilities.monitoring import validate_metrics_field
from utilities.resources.envoy_filter import EnvoyFilter
from utilities.resources.maas_config import Config as MaaSConfig

LOGGER = structlog.get_logger(name=__name__)

MAAS_CONTROLLER_DEPLOYMENT_NAME = "maas-controller"


@contextmanager
def maas_config_usage_logging_enabled(
    maas_config: MaaSConfig,
) -> Generator[MaaSConfig]:
    """Patch Config/default to enable usageLogging and explicitly disable on exit."""
    with ResourceEditor(patches={maas_config: {"spec": {"usageLogging": True}}}):
        yield maas_config
    ResourceEditor(patches={maas_config: {"spec": {"usageLogging": False}}}).update()


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
    for owner_ref in owner_references:
        if owner_ref.uid == config_uid:
            return True
    return False


def config_controller_owner_reference_present(
    owner_references: list[object] | None,
    config_uid: str,
) -> bool:
    """Return True when ownerReferences include Config/default as controller owner."""
    if not owner_references:
        return False
    for owner_ref in owner_references:
        if owner_ref.uid == config_uid and owner_ref.controller is True:
            return True
    return False


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
) -> ServiceMonitor:
    """Wait for maas-controller to create the Limitador ServiceMonitor."""
    service_monitor = ServiceMonitor(
        client=admin_client,
        name=LIMITADOR_SERVICE_MONITOR_NAME,
        namespace=monitoring_namespace,
    )
    try:
        for service_monitor_ready in TimeoutSampler(
            wait_timeout=timeout,
            sleep=5,
            func=lambda: service_monitor.exists,
        ):
            if service_monitor_ready:
                return service_monitor
    except TimeoutExpiredError as exc:
        raise AssertionError(
            f"ServiceMonitor '{LIMITADOR_SERVICE_MONITOR_NAME}' not found in "
            f"'{monitoring_namespace}' after maas-controller reconcile"
        ) from exc
    return service_monitor


def validate_limitador_service_monitor_spec(
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

    namespace_selector = spec.namespaceSelector
    assert namespace_selector.any is True, "Expected namespaceSelector.any=true for cross-namespace Limitador scrape"


def validate_limitador_metrics_in_prometheus(
    prometheus: Prometheus,
    limitador_service_monitor: ServiceMonitor,
    timeout: int = METRICS_POLL_TIMEOUT,
) -> None:
    """Verify Limitador metrics are visible in RHOAI observability Thanos via limitador_up."""
    _ = limitador_service_monitor
    validate_metrics_field(
        prometheus=prometheus,
        metrics_query=limitador_scrape_target_up_query(),
        expected_value="1",
        timeout=timeout,
    )


def usage_logging_resources_present(
    admin_client: DynamicClient,
    monitoring_namespace: str,
) -> bool:
    """Return True when all usage-log observability resources exist."""
    usage_logs_envoy_filter = EnvoyFilter(
        client=admin_client,
        name=USAGE_LOGS_ENVOY_FILTER_NAME,
        namespace=MAAS_GATEWAY_NAMESPACE,
    )
    if not usage_logs_envoy_filter.exists:
        return False

    if not usage_logs_collector_exists(
        admin_client=admin_client,
        namespace=monitoring_namespace,
    ):
        return False

    usage_logs_crb = ClusterRoleBinding(
        client=admin_client,
        name=USAGE_LOGS_CRB_NAME,
    )
    return bool(usage_logs_crb.exists)


def usage_logging_resources_absent(
    admin_client: DynamicClient,
    monitoring_namespace: str,
) -> bool:
    """Return True when all usage-log observability resources are absent."""
    usage_logs_envoy_filter = EnvoyFilter(
        client=admin_client,
        name=USAGE_LOGS_ENVOY_FILTER_NAME,
        namespace=MAAS_GATEWAY_NAMESPACE,
    )
    if usage_logs_envoy_filter.exists:
        return False

    if usage_logs_collector_exists(
        admin_client=admin_client,
        namespace=monitoring_namespace,
    ):
        return False

    usage_logs_crb = ClusterRoleBinding(
        client=admin_client,
        name=USAGE_LOGS_CRB_NAME,
    )
    return not usage_logs_crb.exists


def wait_for_usage_logging_resources(
    admin_client: DynamicClient,
    monitoring_namespace: str,
    timeout: int = USAGE_LOGGING_RESOURCES_WAIT_TIMEOUT,
) -> None:
    """Wait for maas-controller to deploy usage-log observability resources."""
    try:
        for resources_ready in TimeoutSampler(
            wait_timeout=timeout,
            sleep=5,
            func=lambda: usage_logging_resources_present(
                admin_client=admin_client,
                monitoring_namespace=monitoring_namespace,
            ),
        ):
            if resources_ready:
                return
    except TimeoutExpiredError as exc:
        raise AssertionError(
            "Usage-log observability resources not found after enabling Config/default.spec.usageLogging"
        ) from exc


def wait_for_usage_logging_resources_absent(
    admin_client: DynamicClient,
    monitoring_namespace: str,
    timeout: int = USAGE_LOGGING_RESOURCES_WAIT_TIMEOUT,
) -> None:
    """Wait for maas-controller to remove usage-log observability resources."""
    try:
        for resources_removed in TimeoutSampler(
            wait_timeout=timeout,
            sleep=5,
            func=lambda: usage_logging_resources_absent(
                admin_client=admin_client,
                monitoring_namespace=monitoring_namespace,
            ),
        ):
            if resources_removed:
                return
    except TimeoutExpiredError as exc:
        raise AssertionError(
            "Usage-log observability resources still present after disabling Config/default.spec.usageLogging"
        ) from exc


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
    assert_usage_logging_resources_present(
        admin_client=admin_client,
        maas_config=maas_config,
        monitoring_namespace=monitoring_namespace,
    )


def verify_usage_logging_resources_removed(
    admin_client: DynamicClient,
    maas_config: MaaSConfig,
    monitoring_namespace: str,
) -> None:
    """Wait for and validate the usage-log observability stack after usageLogging is disabled."""
    maas_config.get()
    wait_for_usage_logging_resources_absent(
        admin_client=admin_client,
        monitoring_namespace=monitoring_namespace,
    )
    assert_usage_logging_resources_absent(
        admin_client=admin_client,
        maas_config=maas_config,
        monitoring_namespace=monitoring_namespace,
    )


def assert_usage_logging_resources_present(
    admin_client: DynamicClient,
    maas_config: MaaSConfig,
    monitoring_namespace: str,
) -> None:
    """Verify usage-log observability resources exist when usageLogging is enabled."""
    maas_config.get()
    assert usage_logging_enabled(maas_config=maas_config), "Expected Config/default.spec.usageLogging to be enabled"

    config_uid = maas_config.instance.metadata.uid

    usage_logs_envoy_filter = EnvoyFilter(
        client=admin_client,
        name=USAGE_LOGS_ENVOY_FILTER_NAME,
        namespace=MAAS_GATEWAY_NAMESPACE,
        ensure_exists=True,
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
        ensure_exists=True,
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
        ensure_exists=True,
    )
    assert config_controller_owner_reference_present(
        owner_references=usage_logs_crb.instance.metadata.ownerReferences,
        config_uid=config_uid,
    ), f"Expected Config/default UID '{config_uid}' in ClusterRoleBinding '{USAGE_LOGS_CRB_NAME}' ownerReferences"


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

    try:
        for interval_matches in TimeoutSampler(
            wait_timeout=timeout,
            sleep=5,
            func=scrape_interval_matches,
        ):
            if interval_matches:
                return service_monitor
    except TimeoutExpiredError as exc:
        observed_interval = ""
        if service_monitor.exists:
            service_monitor.get()
            observed_interval = str(service_monitor.instance.spec.endpoints[0].interval)
        raise AssertionError(
            f"ServiceMonitor '{LIMITADOR_SERVICE_MONITOR_NAME}' scrape interval "
            f"did not update to '{expected_interval}' (observed '{observed_interval}')"
        ) from exc
    return service_monitor


def verify_limitador_scrape_interval_on_servicemonitor(
    admin_client: DynamicClient,
    maas_config: MaaSConfig,
    monitoring_namespace: str,
    expected_interval: str,
) -> None:
    """Wait for and validate the Limitador ServiceMonitor scrape interval matches Config/default."""
    updated_service_monitor = wait_for_limitador_scrape_interval(
        admin_client=admin_client,
        monitoring_namespace=monitoring_namespace,
        expected_interval=expected_interval,
    )
    maas_config.get()
    validate_limitador_service_monitor_spec(
        limitador_service_monitor=updated_service_monitor,
        maas_config=maas_config,
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

    usage_logs_envoy_filter = EnvoyFilter(
        client=admin_client,
        name=USAGE_LOGS_ENVOY_FILTER_NAME,
        namespace=MAAS_GATEWAY_NAMESPACE,
    )
    assert not usage_logs_envoy_filter.exists, (
        f"EnvoyFilter '{USAGE_LOGS_ENVOY_FILTER_NAME}' must not exist when usageLogging is disabled"
    )

    assert not usage_logs_collector_exists(
        admin_client=admin_client,
        namespace=monitoring_namespace,
    ), f"OpenTelemetryCollector '{USAGE_LOGS_COLLECTOR_NAME}' must not exist when usageLogging is disabled"

    usage_logs_crb = ClusterRoleBinding(
        client=admin_client,
        name=USAGE_LOGS_CRB_NAME,
    )
    assert not usage_logs_crb.exists, (
        f"ClusterRoleBinding '{USAGE_LOGS_CRB_NAME}' must not exist when usageLogging is disabled"
    )
