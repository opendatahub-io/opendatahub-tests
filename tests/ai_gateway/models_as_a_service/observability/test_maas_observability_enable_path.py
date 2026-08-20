import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.service_monitor import ServiceMonitor

from tests.ai_gateway.models_as_a_service.observability.constants import (
    LIMITADOR_SCRAPE_INTERVAL_TEST_VALUE,
)
from tests.ai_gateway.models_as_a_service.observability.utils import (
    maas_config_usage_logging_enabled,
    patch_maas_config_limitador_scrape_interval,
    verify_limitador_scrape_interval_on_servicemonitor,
    verify_usage_logging_resources_deployed,
    verify_usage_logging_resources_removed,
)
from utilities.resources.maas_config import Config as MaaSConfig


@pytest.mark.usefixtures("maas_subscription_controller_enabled_latest")
@pytest.mark.tier2
@pytest.mark.metrics
class TestMaaSObservabilityEnablePath:
    """Tier2 tests for MaaS observability resources reconciled when Config/default features are enabled."""

    def test_usage_logging_resources_exist_when_enabled(
        self,
        admin_client: DynamicClient,
        maas_config_default: MaaSConfig,
        maas_monitoring_namespace: str,
        opentelemetry_collector_crd_available: None,
    ) -> None:
        """Given usageLogging is enabled on Config/default, when maas-controller reconciles,
        then usage-log stack exists.

        Verifies ensureUsageLogs and ensureUsageLogsEnvoyFilter deploy EnvoyFilter, OpenTelemetryCollector,
        and ClusterRoleBinding owned by Config/default. Mirror of the disabled-by-default smoke test.
        """
        with maas_config_usage_logging_enabled(maas_config=maas_config_default):
            verify_usage_logging_resources_deployed(
                admin_client=admin_client,
                maas_config=maas_config_default,
                monitoring_namespace=maas_monitoring_namespace,
            )

    def test_usage_logging_patch_enable_verify_restore(
        self,
        admin_client: DynamicClient,
        maas_config_default: MaaSConfig,
        maas_monitoring_namespace: str,
        opentelemetry_collector_crd_available: None,
    ) -> None:
        """Given Config/default is patched to enable usageLogging, when resources reconcile and Config is restored,
        then the usage-log stack is removed again.

        Verifies ensureUsageLogs and ensureUsageLogsEnvoyFilter tear down owned resources when usageLogging
        is disabled after a prior enable reconcile.
        """
        with maas_config_usage_logging_enabled(maas_config=maas_config_default):
            verify_usage_logging_resources_deployed(
                admin_client=admin_client,
                maas_config=maas_config_default,
                monitoring_namespace=maas_monitoring_namespace,
            )

        verify_usage_logging_resources_removed(
            admin_client=admin_client,
            maas_config=maas_config_default,
            monitoring_namespace=maas_monitoring_namespace,
        )

    def test_limitador_scrape_interval_applied_to_servicemonitor(
        self,
        admin_client: DynamicClient,
        maas_config_default: MaaSConfig,
        maas_monitoring_namespace: str,
        limitador_service_monitor: ServiceMonitor,
    ) -> None:
        """Given limitadorScrapeInterval is set on Config/default, when maas-controller reconciles,
        then the Limitador ServiceMonitor uses that interval.

        Verifies ensureLimitadorServiceMonitor honors Config/default.spec.limitadorScrapeInterval.
        Config is restored when the test completes.
        """
        assert limitador_service_monitor.exists, (
            f"ServiceMonitor '{limitador_service_monitor.name}' must exist before scrape interval patch"
        )
        with patch_maas_config_limitador_scrape_interval(
            maas_config=maas_config_default,
            scrape_interval=LIMITADOR_SCRAPE_INTERVAL_TEST_VALUE,
        ):
            verify_limitador_scrape_interval_on_servicemonitor(
                admin_client=admin_client,
                maas_config=maas_config_default,
                monitoring_namespace=maas_monitoring_namespace,
                expected_interval=LIMITADOR_SCRAPE_INTERVAL_TEST_VALUE,
            )
