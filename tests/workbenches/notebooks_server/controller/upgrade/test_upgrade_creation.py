import pytest
from ocp_resources.notebook import Notebook
from ocp_resources.pod import Pod
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.service import Service

from tests.workbenches.notebooks_server.controller.utils import StatefulSet

OAUTH_PROXY_CONTAINER = "oauth-proxy"
OAUTH_PROXY_TLS_PORT = 443


class TestPostUpgradeNotebookCreation:
    """Verify a new notebook can be created on the upgraded platform.

    Steps:
        1. Create a fresh Notebook CR on the upgraded controller.
        2. Verify the pod reaches Ready state.
        3. Verify the StatefulSet and Service are created.
        4. Verify the Route is created targeting the TLS Service.
        5. Verify the oauth-proxy sidecar is injected.
        6. Verify auth resources (TLS Service, oauth-config Secret, TLS Secret) are reconciled.
        7. Verify the reconciliation lock annotation is cleared.
    """

    @pytest.mark.post_upgrade
    def test_new_notebook_pod_ready(
        self,
        new_notebook_pod: Pod,
    ) -> None:
        """Given the platform was upgraded,
        When a new Notebook CR is created,
        Then the notebook pod should reach Ready state.

        Validation is performed by the new_notebook_pod fixture which waits
        for the pod to exist and reach Ready condition.
        """

    @pytest.mark.post_upgrade
    def test_new_notebook_statefulset_exists(
        self,
        new_notebook_statefulset: StatefulSet,
    ) -> None:
        """Given a new notebook is created post-upgrade,
        When the controller reconciles,
        Then a StatefulSet should be created with 1 replica.
        """
        assert new_notebook_statefulset.exists, (
            f"StatefulSet '{new_notebook_statefulset.name}' was not created on upgraded platform"
        )

        replicas = new_notebook_statefulset.instance.spec.replicas
        assert replicas == 1, f"StatefulSet has {replicas} replicas, expected 1"

    @pytest.mark.post_upgrade
    def test_new_notebook_service_exists(
        self,
        new_notebook_service: Service,
    ) -> None:
        """Given a new notebook is created post-upgrade,
        When the controller reconciles,
        Then a Service should be created.
        """
        assert new_notebook_service.exists, (
            f"Service '{new_notebook_service.name}' was not created on upgraded platform"
        )

    @pytest.mark.post_upgrade
    def test_new_notebook_route_exists(
        self,
        new_notebook: Notebook,
        new_notebook_route: Route,
    ) -> None:
        """Given a new notebook is created post-upgrade,
        When the controller reconciles routing,
        Then a Route should be created targeting the TLS Service.
        """
        assert new_notebook_route.exists, f"Route '{new_notebook_route.name}' was not created on upgraded platform"

        expected_service = f"{new_notebook.name}-tls"
        backend_service = new_notebook_route.exposed_service
        assert backend_service == expected_service, (
            f"Route '{new_notebook_route.name}' targets service '{backend_service}', expected '{expected_service}'"
        )

    @pytest.mark.post_upgrade
    def test_new_notebook_has_oauth_proxy_sidecar(
        self,
        new_notebook_pod: Pod,
    ) -> None:
        """Given a new notebook is created post-upgrade with inject-oauth,
        When the controller mutates the CR,
        Then the pod should have an oauth-proxy sidecar container.
        """
        containers = new_notebook_pod.instance.spec.containers
        container_names = [container.name for container in containers]

        assert OAUTH_PROXY_CONTAINER in container_names, (
            f"Pod '{new_notebook_pod.name}' missing '{OAUTH_PROXY_CONTAINER}' sidecar on upgraded platform. "
            f"Containers: {container_names}"
        )

    @pytest.mark.post_upgrade
    def test_new_notebook_tls_service_exists(
        self,
        new_notebook_tls_service: Service,
    ) -> None:
        """Given a new notebook is created post-upgrade with inject-oauth,
        When the controller reconciles auth resources,
        Then a TLS Service should be created with port 443.
        """
        assert new_notebook_tls_service.exists, (
            f"TLS Service '{new_notebook_tls_service.name}' was not created on upgraded platform"
        )

        port_numbers = [port.port for port in new_notebook_tls_service.instance.spec.ports]
        assert OAUTH_PROXY_TLS_PORT in port_numbers, (
            f"Service '{new_notebook_tls_service.name}' missing port {OAUTH_PROXY_TLS_PORT}. "
            f"Found ports: {port_numbers}"
        )

    @pytest.mark.post_upgrade
    def test_new_notebook_oauth_config_secret_exists(
        self,
        new_notebook_oauth_config_secret: Secret,
    ) -> None:
        """Given a new notebook is created post-upgrade with inject-oauth,
        When the controller reconciles auth resources,
        Then an oauth-config Secret should be created.
        """
        assert new_notebook_oauth_config_secret.exists, (
            f"oauth-config Secret '{new_notebook_oauth_config_secret.name}' was not created on upgraded platform"
        )

    @pytest.mark.post_upgrade
    def test_new_notebook_tls_secret_exists(
        self,
        new_notebook_tls_secret: Secret,
    ) -> None:
        """Given a new notebook is created post-upgrade with inject-oauth,
        When the controller reconciles auth resources,
        Then a TLS Secret should be created.
        """
        assert new_notebook_tls_secret.exists, (
            f"TLS Secret '{new_notebook_tls_secret.name}' was not created on upgraded platform"
        )

    @pytest.mark.post_upgrade
    def test_new_notebook_reconciliation_lock_cleared(
        self,
        new_notebook: Notebook,
    ) -> None:
        """Given a new notebook is created post-upgrade,
        When the ODH controller completes its first reconciliation,
        Then the reconciliation lock annotation should be cleared.
        """
        stop_annotation = new_notebook.instance.metadata.annotations.get("kubeflow-resource-stopped")
        assert stop_annotation != "odh-notebook-controller-lock", (
            f"Notebook '{new_notebook.name}' still has reconciliation lock "
            f"'odh-notebook-controller-lock' after pod reached Ready"
        )
