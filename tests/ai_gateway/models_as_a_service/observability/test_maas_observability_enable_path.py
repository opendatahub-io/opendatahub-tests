import pytest
from kubernetes.dynamic import DynamicClient

from tests.ai_gateway.models_as_a_service.observability.utils import (
    expected_limitador_scrape_interval,
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

    @pytest.mark.parametrize(
        "usage_logging_state",
        [
            pytest.param(
                "enabled",
                marks=pytest.mark.dependency(name="test_usage_logging_resources_exist_when_enabled"),
                id="test_usage_logging_resources_exist_when_enabled",
            ),
            pytest.param(
                "disabled",
                marks=pytest.mark.dependency(depends=["test_usage_logging_resources_exist_when_enabled"]),
                id="test_usage_logging_patch_enable_verify_restore",
            ),
        ],
    )
    def test_usage_logging_resources_reconcile(
        self,
        usage_logging_state: str,
        admin_client: DynamicClient,
        maas_config_default: MaaSConfig,
        maas_monitoring_namespace: str,
        opentelemetry_collector_crd_available: None,
        request: pytest.FixtureRequest,
    ) -> None:
        """Given usageLogging is toggled on Config/default, when maas-controller reconciles,
        then usage-log observability resources are deployed or removed accordingly.

        Verifies ensureUsageLogs and ensureUsageLogsEnvoyFilter deploy owned resources when enabled
        and tear them down after usageLogging is restored to disabled.
        """
        if usage_logging_state == "enabled":
            maas_config = request.getfixturevalue(argname="maas_config_with_usage_logging_enabled")
            verify_usage_logging_resources_deployed(
                admin_client=admin_client,
                maas_config=maas_config,
                monitoring_namespace=maas_monitoring_namespace,
            )
        else:
            verify_usage_logging_resources_removed(
                admin_client=admin_client,
                maas_config=maas_config_default,
                monitoring_namespace=maas_monitoring_namespace,
            )

    @pytest.mark.usefixtures("limitador_service_monitor")
    @pytest.mark.parametrize(
        "limitador_scrape_interval_state",
        [
            pytest.param(
                "patched",
                marks=pytest.mark.dependency(name="test_limitador_scrape_interval_applied_to_servicemonitor"),
                id="test_limitador_scrape_interval_applied_to_servicemonitor",
            ),
            pytest.param(
                "restored",
                marks=pytest.mark.dependency(depends=["test_limitador_scrape_interval_applied_to_servicemonitor"]),
                id="test_limitador_scrape_interval_restored_on_config_default",
            ),
        ],
    )
    def test_limitador_scrape_interval_reconcile(
        self,
        limitador_scrape_interval_state: str,
        admin_client: DynamicClient,
        maas_config_default: MaaSConfig,
        maas_monitoring_namespace: str,
        request: pytest.FixtureRequest,
    ) -> None:
        """Given limitadorScrapeInterval is set on Config/default, when maas-controller reconciles,
        then the Limitador ServiceMonitor uses that interval and restores after Config is reset.

        Verifies ensureLimitadorServiceMonitor honors Config/default.spec.limitadorScrapeInterval.
        """
        if limitador_scrape_interval_state == "patched":
            maas_config = request.getfixturevalue(argname="maas_config_with_limitador_scrape_interval_patched")
            verify_limitador_scrape_interval_on_servicemonitor(
                admin_client=admin_client,
                monitoring_namespace=maas_monitoring_namespace,
                expected_interval=expected_limitador_scrape_interval(maas_config=maas_config),
            )
        else:
            verify_limitador_scrape_interval_on_servicemonitor(
                admin_client=admin_client,
                monitoring_namespace=maas_monitoring_namespace,
                expected_interval=expected_limitador_scrape_interval(maas_config=maas_config_default),
            )
