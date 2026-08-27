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
from pytest_testconfig import config as py_config
from timeout_sampler import TimeoutSampler

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
    OTEL_COLLECTOR_CRD_NAME,
    USAGE_LOGGING_RESOURCES_WAIT_TIMEOUT,
    USAGE_LOGS_COLLECTOR_NAME,
    USAGE_LOGS_CRB_NAME,
    USAGE_LOGS_ENVOY_FILTER_NAME,
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
