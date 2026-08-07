import structlog
from ocp_resources.route import Route

LOGGER = structlog.get_logger(name=__name__)


def test_openshell_release_installed(installed_openshell_release: str) -> None:
    """
    Exercises the full OpenShell Helm install fixture: namespace + SCC setup, OCI chart
    install, and gateway pod readiness wait.
    """
    route_host = installed_openshell_release
    LOGGER.info(f"OpenShell installed, route host: {route_host}")
    assert route_host


def test_openshell_gateway_route(openshell_gateway_route: Route) -> None:
    """
    Exercises the passthrough Route fixture exposing the OpenShell gateway.
    """
    assert openshell_gateway_route.exists
    assert openshell_gateway_route.instance.spec.tls.termination == "passthrough"
