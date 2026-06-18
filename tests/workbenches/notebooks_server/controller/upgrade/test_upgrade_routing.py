import json
from typing import Any

import pytest
from ocp_resources.notebook import Notebook
from ocp_resources.route import Route
from ocp_resources.service import Service


@pytest.mark.usefixtures("capture_notebook_baseline")
class TestPreUpgradeNotebookRouting:
    """Verify notebook routing resources (OpenShift Route) exist before the platform upgrade.

    Steps:
        1. Verify the Route for the notebook exists in the notebook namespace.
        2. Verify the Route targets the correct backend TLS Service.
        3. Verify the Route has TLS termination configured.
        4. Verify the TLS Service exists.
    """

    @pytest.mark.pre_upgrade
    def test_route_exists_before_upgrade(
        self,
        upgrade_notebook_route: Route,
    ) -> None:
        """Given a Notebook CR is created before upgrade,
        When the notebook controller reconciles routing,
        Then a Route should exist for the notebook in the notebook namespace.
        """
        assert upgrade_notebook_route.exists, (
            f"Route '{upgrade_notebook_route.name}' does not exist in namespace '{upgrade_notebook_route.namespace}'"
        )

    @pytest.mark.pre_upgrade
    def test_route_targets_correct_service_before_upgrade(
        self,
        upgrade_notebook: Notebook,
        upgrade_notebook_route: Route,
    ) -> None:
        """Given the notebook Route exists,
        When inspecting its spec.to,
        Then it should target the notebook's TLS Service (oauth-proxy creates a -tls suffixed service).
        """
        expected_service = f"{upgrade_notebook.name}-tls"
        backend_service = upgrade_notebook_route.exposed_service
        assert backend_service == expected_service, (
            f"Route '{upgrade_notebook_route.name}' targets service '{backend_service}', expected '{expected_service}'"
        )

    @pytest.mark.pre_upgrade
    def test_tls_service_exists_before_upgrade(
        self,
        upgrade_notebook_tls_service: Service,
    ) -> None:
        """Given a Notebook CR is created before upgrade,
        When the notebook controller reconciles,
        Then a TLS Service (name-tls) should exist for the oauth-proxy port.
        """
        assert upgrade_notebook_tls_service.exists, (
            f"TLS Service '{upgrade_notebook_tls_service.name}' does not exist "
            f"in namespace '{upgrade_notebook_tls_service.namespace}'"
        )

    @pytest.mark.pre_upgrade
    def test_route_has_tls_before_upgrade(
        self,
        upgrade_notebook_route: Route,
    ) -> None:
        """Given the notebook Route exists,
        When inspecting its TLS configuration,
        Then TLS termination should be configured.
        """
        tls = upgrade_notebook_route.instance.spec.get("tls")
        assert tls, f"Route '{upgrade_notebook_route.name}' has no TLS configuration"

        termination = tls.get("termination")
        assert termination, f"Route '{upgrade_notebook_route.name}' has TLS config but no termination mode set"


class TestPostUpgradeNotebookRouting:
    """Verify notebook routing survived the platform upgrade.

    Steps:
        1. Verify Route still exists.
        2. Verify Route spec was not modified (generation unchanged).
        3. Verify Route still targets the correct backend Service.
        4. Verify Route hostname was preserved.
        5. Verify Route TLS configuration was preserved.
        6. Verify TLS Service was not modified.
    """

    @pytest.mark.post_upgrade
    def test_route_exists_after_upgrade(
        self,
        upgrade_notebook_route: Route,
    ) -> None:
        """Given a notebook Route existed before upgrade,
        When the upgrade completes,
        Then the Route should still exist.
        """
        assert upgrade_notebook_route.exists, f"Route '{upgrade_notebook_route.name}' no longer exists after upgrade"

    @pytest.mark.post_upgrade
    def test_route_not_modified_after_upgrade(
        self,
        upgrade_notebook_route: Route,
        upgrade_notebook_baseline: dict[str, Any],
    ) -> None:
        """Given a notebook Route existed before upgrade,
        When the upgrade completes,
        Then the Route generation should be unchanged.
        """
        assert upgrade_notebook_route.exists, f"Route '{upgrade_notebook_route.name}' no longer exists after upgrade"
        current_generation = upgrade_notebook_route.instance.metadata.generation
        saved_generation = upgrade_notebook_baseline["route_generation"]

        assert current_generation == saved_generation, (
            f"Route was modified during upgrade. "
            f"Pre-upgrade generation: {saved_generation}, "
            f"post-upgrade generation: {current_generation}"
        )

    @pytest.mark.post_upgrade
    def test_route_targets_correct_service_after_upgrade(
        self,
        upgrade_notebook: Notebook,
        upgrade_notebook_route: Route,
    ) -> None:
        """Given a notebook Route existed before upgrade,
        When the upgrade completes,
        Then it should still target the notebook's TLS Service.
        """
        assert upgrade_notebook_route.exists, f"Route '{upgrade_notebook_route.name}' no longer exists after upgrade"
        expected_service = f"{upgrade_notebook.name}-tls"
        backend_service = upgrade_notebook_route.exposed_service
        assert backend_service == expected_service, (
            f"Route '{upgrade_notebook_route.name}' targets service '{backend_service}' after upgrade, "
            f"expected '{expected_service}'"
        )

    @pytest.mark.post_upgrade
    def test_route_host_preserved_after_upgrade(
        self,
        upgrade_notebook_route: Route,
        upgrade_notebook_baseline: dict[str, Any],
    ) -> None:
        """Given a notebook Route existed before upgrade,
        When the upgrade completes,
        Then the Route hostname should be unchanged.
        """
        assert upgrade_notebook_route.exists, f"Route '{upgrade_notebook_route.name}' no longer exists after upgrade"
        current_host = upgrade_notebook_route.host
        saved_host = upgrade_notebook_baseline["route_host"]

        assert current_host == saved_host, (
            f"Route hostname changed during upgrade. Pre-upgrade: '{saved_host}', post-upgrade: '{current_host}'"
        )

    @pytest.mark.post_upgrade
    def test_route_tls_preserved_after_upgrade(
        self,
        upgrade_notebook_route: Route,
        upgrade_notebook_baseline: dict[str, Any],
    ) -> None:
        """Given a notebook Route existed before upgrade,
        When the upgrade completes,
        Then the TLS termination mode should be unchanged.
        """
        assert upgrade_notebook_route.exists, f"Route '{upgrade_notebook_route.name}' no longer exists after upgrade"
        tls = upgrade_notebook_route.instance.spec.get("tls", {})
        current_termination = tls.get("termination", "")
        saved_termination = upgrade_notebook_baseline["route_tls_termination"]

        assert current_termination == saved_termination, (
            f"Route TLS termination changed during upgrade. "
            f"Pre-upgrade: '{saved_termination}', post-upgrade: '{current_termination}'"
        )

    @pytest.mark.post_upgrade
    def test_tls_service_not_modified_after_upgrade(
        self,
        upgrade_notebook_tls_service: Service,
        upgrade_notebook_baseline: dict[str, Any],
    ) -> None:
        """Given a notebook TLS Service existed before upgrade,
        When the upgrade completes,
        Then the TLS Service spec (ports, selector) should be unchanged.
        """
        assert upgrade_notebook_tls_service.exists, (
            f"TLS Service '{upgrade_notebook_tls_service.name}' no longer exists after upgrade"
        )

        tls_service_spec = upgrade_notebook_tls_service.instance.spec
        current_ports = json.dumps(tls_service_spec.ports, sort_keys=True, default=str)
        current_selector = json.dumps(tls_service_spec.selector, sort_keys=True, default=str)

        saved_ports = upgrade_notebook_baseline["tls_service_ports"]
        saved_selector = upgrade_notebook_baseline["tls_service_selector"]

        assert current_ports == saved_ports, (
            f"TLS Service ports were modified during upgrade. Pre-upgrade: {saved_ports}, post-upgrade: {current_ports}"
        )

        assert current_selector == saved_selector, (
            f"TLS Service selector was modified during upgrade. "
            f"Pre-upgrade: {saved_selector}, post-upgrade: {current_selector}"
        )
