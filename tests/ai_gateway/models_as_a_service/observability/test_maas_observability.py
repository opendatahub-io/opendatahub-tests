import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.service_monitor import ServiceMonitor
from ocp_utilities.monitoring import Prometheus

from tests.ai_gateway.models_as_a_service.observability.constants import (
    METRICS_POLL_TIMEOUT,
    limitador_scrape_target_up_query,
)
from tests.ai_gateway.models_as_a_service.observability.utils import (
    assert_usage_logging_resources_absent,
    validate_limitador_service_monitor_spec,
)
from utilities.monitoring import validate_metrics_field
from utilities.resources.maas_config import Config as MaaSConfig


@pytest.mark.usefixtures("maas_subscription_controller_enabled_latest")
@pytest.mark.smoke
@pytest.mark.metrics
class TestMaaSObservability:
    """Smoke tests for MaaS platform observability reconciled by maas-controller."""

    def test_limitador_servicemonitor_exists(
        self,
        admin_client: DynamicClient,
        maas_config_default: MaaSConfig,
        limitador_service_monitor: ServiceMonitor,
    ) -> None:
        """Given MaaS is managed, when maas-controller reconciles observability, then Limitador ServiceMonitor exists.

        Verifies the ServiceMonitor bootstrapped by ensureLimitadorServiceMonitor with the expected
        scrape configuration and Config/default owner reference.
        """
        validate_limitador_service_monitor_spec(
            admin_client=admin_client,
            limitador_service_monitor=limitador_service_monitor,
            maas_config=maas_config_default,
        )

    def test_limitador_metrics_in_prometheus(
        self,
        limitador_service_monitor: ServiceMonitor,
        limitador_deployed: None,
        maas_observability_prometheus: Prometheus,
    ) -> None:
        """Given Limitador is running and ServiceMonitor exists, when querying RHOAI Thanos,
        then limitador_up is scraped.

        Verifies the end-to-end metrics path from Limitador through the MaaS-managed ServiceMonitor
        into the RHOAI observability stack.
        """
        validate_metrics_field(
            prometheus=maas_observability_prometheus,
            metrics_query=limitador_scrape_target_up_query(),
            expected_value="1",
            timeout=METRICS_POLL_TIMEOUT,
        )

    def test_usage_logging_disabled_by_default(
        self,
        admin_client: DynamicClient,
        maas_config_default: MaaSConfig,
        maas_monitoring_namespace: str,
    ) -> None:
        """Given a fresh MaaS install, when usageLogging is disabled on Config/default, then usage-log stack is absent.

        Verifies ensureUsageLogs and ensureUsageLogsEnvoyFilter do not leave observability resources deployed
        when the usageLogging feature gate is off.
        """
        assert_usage_logging_resources_absent(
            admin_client=admin_client,
            maas_config=maas_config_default,
            monitoring_namespace=maas_monitoring_namespace,
        )
