import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.custom_resource_definition import CustomResourceDefinition
from ocp_resources.route import Route
from ocp_resources.service_monitor import ServiceMonitor
from ocp_utilities.monitoring import Prometheus

from tests.ai_gateway.models_as_a_service.observability.constants import (
    LIMITADOR_SERVICE_MONITOR_NAME,
    RHOAI_THANOS_QUERIER_ROUTE_NAME,
    SERVICE_MONITOR_CRD_NAME,
)
from tests.ai_gateway.models_as_a_service.observability.utils import (
    get_maas_config_default,
    limitador_is_deployed,
    monitoring_namespace_exists,
    resolve_maas_monitoring_namespace,
    wait_for_limitador_service_monitor,
)
from utilities.certificates_utils import create_ca_bundle_file
from utilities.infra import get_openshift_token
from utilities.resources.maas_config import Config as MaaSConfig


@pytest.fixture(scope="session")
def maas_monitoring_namespace(
    admin_client: DynamicClient,
    maas_subscription_controller_enabled_latest: None,
) -> str:
    """Return the monitoring namespace configured for maas-controller observability."""
    monitoring_namespace = resolve_maas_monitoring_namespace(admin_client=admin_client)
    if not monitoring_namespace:
        pytest.fail("maas-controller MONITORING_NAMESPACE is not configured; observability reconcile cannot run")
    if not monitoring_namespace_exists(admin_client=admin_client, namespace_name=monitoring_namespace):
        pytest.fail(
            f"Monitoring namespace '{monitoring_namespace}' does not exist; "
            "maas-controller observability setup requires it"
        )
    return monitoring_namespace


@pytest.fixture(scope="session")
def servicemonitor_crd_available(admin_client: DynamicClient) -> None:
    """Fail when the Prometheus Operator ServiceMonitor CRD is not installed."""
    service_monitor_crd = CustomResourceDefinition(
        client=admin_client,
        name=SERVICE_MONITOR_CRD_NAME,
    )
    if not service_monitor_crd.exists:
        pytest.fail(
            f"ServiceMonitor CRD '{SERVICE_MONITOR_CRD_NAME}' not installed; Limitador metrics scrape requires it"
        )


@pytest.fixture(scope="session")
def maas_config_default(
    admin_client: DynamicClient,
    maas_subscription_controller_enabled_latest: None,
) -> MaaSConfig:
    """Return the cluster-scoped MaaS Config/default anchor."""
    return get_maas_config_default(admin_client=admin_client)


@pytest.fixture(scope="session")
def limitador_service_monitor(
    admin_client: DynamicClient,
    maas_monitoring_namespace: str,
    servicemonitor_crd_available: None,
) -> ServiceMonitor:
    """Wait for maas-controller to create the Limitador ServiceMonitor."""
    limitador_service_monitor = wait_for_limitador_service_monitor(
        admin_client=admin_client,
        monitoring_namespace=maas_monitoring_namespace,
    )
    if limitador_service_monitor is None:
        pytest.fail(
            f"ServiceMonitor '{LIMITADOR_SERVICE_MONITOR_NAME}' not found in "
            f"'{maas_monitoring_namespace}' after maas-controller reconcile"
        )
    return limitador_service_monitor


@pytest.fixture(scope="session")
def maas_observability_prometheus(
    admin_client: DynamicClient,
    maas_monitoring_namespace: str,
) -> Prometheus:
    """Return a Prometheus client for the RHOAI observability Thanos querier route."""
    thanos_route = Route(
        client=admin_client,
        name=RHOAI_THANOS_QUERIER_ROUTE_NAME,
        namespace=maas_monitoring_namespace,
    )
    if not thanos_route.exists:
        pytest.fail(
            f"Route '{RHOAI_THANOS_QUERIER_ROUTE_NAME}' not found in '{maas_monitoring_namespace}'; "
            "RHOAI observability stack is not deployed"
        )
    return Prometheus(
        client=admin_client,
        namespace=maas_monitoring_namespace,
        resource_name=RHOAI_THANOS_QUERIER_ROUTE_NAME,
        verify_ssl=create_ca_bundle_file(client=admin_client),
        bearer_token=get_openshift_token(),
    )


@pytest.fixture(scope="session")
def limitador_deployed(
    admin_client: DynamicClient,
) -> None:
    """Fail when Limitador is not running in a known policy-engine namespace."""
    if not limitador_is_deployed(admin_client=admin_client):
        pytest.fail(
            "Limitador is not deployed in kuadrant-system or rh-connectivity-link; "
            "Limitador Prometheus metrics smoke test requires it"
        )
