"""OpenShell install-validation tests.

These verify that the Helm-based install fixtures produce a working
OpenShell deployment and gateway Route.  They are tagged ``tier1`` (not
``smoke``) because they validate infrastructure rather than product
behaviour.
"""

import pytest
import structlog
from ocp_resources.route import Route

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.open_shell
@pytest.mark.tier1
def test_openshell_release_installed(installed_openshell_release: str) -> None:
    """Exercises the full OpenShell Helm install fixture: namespace + SCC setup,
    OCI chart install, and gateway pod readiness wait.
    """
    route_host = installed_openshell_release
    LOGGER.info("OpenShell installed", route_host=route_host)
    assert route_host


@pytest.mark.open_shell
@pytest.mark.tier1
def test_openshell_gateway_route(openshell_gateway_route: Route) -> None:
    """Exercises the passthrough Route fixture exposing the OpenShell gateway."""
    assert openshell_gateway_route.exists
    assert openshell_gateway_route.instance.spec.tls.termination == "passthrough"
