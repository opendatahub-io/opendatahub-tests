import pytest
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.service import Service

OAUTH_PROXY_CONTAINER = "oauth-proxy"
OAUTH_PROXY_TLS_PORT = 443


@pytest.mark.usefixtures("capture_notebook_baseline")
class TestPreUpgradeNotebookAuth:
    """Verify oauth-proxy auth resources exist before the platform upgrade.

    Steps:
        1. Verify the notebook pod has an oauth-proxy sidecar container.
        2. Verify the TLS Service exists with port 443.
        3. Verify the oauth-config Secret exists.
        4. Verify the TLS Secret exists.
    """

    @pytest.mark.pre_upgrade
    def test_oauth_proxy_sidecar_present(
        self,
        upgrade_notebook_pod: Pod,
    ) -> None:
        """Given a notebook with inject-oauth annotation,
        When the notebook controller injects the sidecar,
        Then the pod should have an oauth-proxy container.
        """
        containers = upgrade_notebook_pod.instance.spec.containers
        container_names = [container.name for container in containers]

        assert OAUTH_PROXY_CONTAINER in container_names, (
            f"Pod '{upgrade_notebook_pod.name}' missing '{OAUTH_PROXY_CONTAINER}' sidecar. "
            f"Containers: {container_names}"
        )

    @pytest.mark.pre_upgrade
    def test_tls_service_exists(
        self,
        upgrade_notebook_tls_service: Service,
    ) -> None:
        """Given a notebook with inject-oauth annotation,
        When the controller reconciles,
        Then a TLS Service should exist with port 443.
        """
        assert upgrade_notebook_tls_service.exists, f"TLS Service '{upgrade_notebook_tls_service.name}' does not exist"

        port_numbers = [port.port for port in upgrade_notebook_tls_service.instance.spec.ports]
        assert OAUTH_PROXY_TLS_PORT in port_numbers, (
            f"Service '{upgrade_notebook_tls_service.name}' missing port {OAUTH_PROXY_TLS_PORT}. "
            f"Found ports: {port_numbers}"
        )

    @pytest.mark.pre_upgrade
    def test_oauth_config_secret_exists(
        self,
        auth_oauth_config_secret: Secret,
    ) -> None:
        """Given a notebook with inject-oauth annotation,
        When the controller reconciles,
        Then the oauth-config Secret should exist.
        """
        assert auth_oauth_config_secret.exists, f"oauth-config Secret '{auth_oauth_config_secret.name}' does not exist"

    @pytest.mark.pre_upgrade
    def test_tls_secret_exists(
        self,
        auth_tls_secret: Secret,
    ) -> None:
        """Given a notebook with inject-oauth annotation,
        When the controller reconciles,
        Then the TLS Secret should exist.
        """
        assert auth_tls_secret.exists, f"TLS Secret '{auth_tls_secret.name}' does not exist"


@pytest.mark.usefixtures("stopped_notebook_pre_upgrade_shutdown")
class TestPreUpgradeStoppedNotebookAuth:
    """Verify oauth-proxy auth resources exist for a stopped notebook before upgrade.

    Steps:
        1. Verify the TLS Service exists for the stopped notebook.
        2. Verify the oauth-config Secret exists for the stopped notebook.
        3. Verify the TLS Secret exists for the stopped notebook.
    """

    @pytest.mark.pre_upgrade
    def test_stopped_tls_service_exists(
        self,
        stopped_tls_service: Service,
    ) -> None:
        """Given a stopped notebook with inject-oauth annotation,
        When the notebook is stopped,
        Then the TLS Service should still exist with port 443.
        """
        assert stopped_tls_service.exists, (
            f"TLS Service '{stopped_tls_service.name}' does not exist for stopped notebook"
        )

        port_numbers = [port.port for port in stopped_tls_service.instance.spec.ports]
        assert OAUTH_PROXY_TLS_PORT in port_numbers, (
            f"Service '{stopped_tls_service.name}' missing port {OAUTH_PROXY_TLS_PORT}. Found ports: {port_numbers}"
        )

    @pytest.mark.pre_upgrade
    def test_stopped_oauth_config_secret_exists(
        self,
        stopped_auth_oauth_config_secret: Secret,
    ) -> None:
        """Given a stopped notebook with inject-oauth annotation,
        When the notebook is stopped,
        Then the oauth-config Secret should still exist.
        """
        assert stopped_auth_oauth_config_secret.exists, (
            f"oauth-config Secret '{stopped_auth_oauth_config_secret.name}' does not exist for stopped notebook"
        )

    @pytest.mark.pre_upgrade
    def test_stopped_tls_secret_exists(
        self,
        stopped_auth_tls_secret: Secret,
    ) -> None:
        """Given a stopped notebook with inject-oauth annotation,
        When the notebook is stopped,
        Then the TLS Secret should still exist.
        """
        assert stopped_auth_tls_secret.exists, (
            f"TLS Secret '{stopped_auth_tls_secret.name}' does not exist for stopped notebook"
        )


class TestPostUpgradeNotebookAuth:
    """Verify oauth-proxy auth resources survived the platform upgrade.

    Steps:
        1. Verify the sidecar container is still present in the running notebook pod.
        2. Verify the TLS Service still exists for both running and stopped notebooks.
        3. Verify the oauth-config Secret still exists for both running and stopped notebooks.
        4. Verify the TLS Secret still exists for both running and stopped notebooks.
    """

    @pytest.mark.post_upgrade
    def test_oauth_proxy_sidecar_present_after_upgrade(
        self,
        upgrade_notebook_pod: Pod,
    ) -> None:
        """Given a notebook with oauth-proxy sidecar existed before upgrade,
        When the upgrade completes,
        Then the pod should still have the oauth-proxy container.
        """
        containers = upgrade_notebook_pod.instance.spec.containers
        container_names = [container.name for container in containers]

        assert OAUTH_PROXY_CONTAINER in container_names, (
            f"Pod '{upgrade_notebook_pod.name}' lost '{OAUTH_PROXY_CONTAINER}' sidecar after upgrade. "
            f"Containers: {container_names}"
        )

    @pytest.mark.post_upgrade
    def test_tls_service_exists_after_upgrade(
        self,
        upgrade_notebook_tls_service: Service,
    ) -> None:
        """Given a TLS Service existed before upgrade,
        When the upgrade completes,
        Then the Service should still exist with port 443.
        """
        assert upgrade_notebook_tls_service.exists, (
            f"TLS Service '{upgrade_notebook_tls_service.name}' no longer exists after upgrade"
        )

        port_numbers = [port.port for port in upgrade_notebook_tls_service.instance.spec.ports]
        assert OAUTH_PROXY_TLS_PORT in port_numbers, (
            f"Service '{upgrade_notebook_tls_service.name}' lost port {OAUTH_PROXY_TLS_PORT} after upgrade. "
            f"Found ports: {port_numbers}"
        )

    @pytest.mark.post_upgrade
    def test_oauth_config_secret_exists_after_upgrade(
        self,
        auth_oauth_config_secret: Secret,
    ) -> None:
        """Given an oauth-config Secret existed before upgrade,
        When the upgrade completes,
        Then the Secret should still exist.
        """
        assert auth_oauth_config_secret.exists, (
            f"oauth-config Secret '{auth_oauth_config_secret.name}' no longer exists after upgrade"
        )

    @pytest.mark.post_upgrade
    def test_tls_secret_exists_after_upgrade(
        self,
        auth_tls_secret: Secret,
    ) -> None:
        """Given a TLS Secret existed before upgrade,
        When the upgrade completes,
        Then the Secret should still exist.
        """
        assert auth_tls_secret.exists, f"TLS Secret '{auth_tls_secret.name}' no longer exists after upgrade"

    @pytest.mark.post_upgrade
    def test_stopped_tls_service_exists_after_upgrade(
        self,
        stopped_tls_service: Service,
    ) -> None:
        """Given a stopped notebook's TLS Service existed before upgrade,
        When the upgrade completes,
        Then the Service should still exist with port 443 despite the notebook being stopped.
        """
        assert stopped_tls_service.exists, (
            f"TLS Service '{stopped_tls_service.name}' no longer exists after upgrade for stopped notebook"
        )

        port_numbers = [port.port for port in stopped_tls_service.instance.spec.ports]
        assert OAUTH_PROXY_TLS_PORT in port_numbers, (
            f"Service '{stopped_tls_service.name}' lost port {OAUTH_PROXY_TLS_PORT} after upgrade. "
            f"Found ports: {port_numbers}"
        )

    @pytest.mark.post_upgrade
    def test_stopped_oauth_config_secret_exists_after_upgrade(
        self,
        stopped_auth_oauth_config_secret: Secret,
    ) -> None:
        """Given a stopped notebook's oauth-config Secret existed before upgrade,
        When the upgrade completes,
        Then the Secret should still exist despite the notebook being stopped.
        """
        assert stopped_auth_oauth_config_secret.exists, (
            f"oauth-config Secret '{stopped_auth_oauth_config_secret.name}' "
            f"no longer exists after upgrade for stopped notebook"
        )

    @pytest.mark.post_upgrade
    def test_stopped_tls_secret_exists_after_upgrade(
        self,
        stopped_auth_tls_secret: Secret,
    ) -> None:
        """Given a stopped notebook's TLS Secret existed before upgrade,
        When the upgrade completes,
        Then the Secret should still exist despite the notebook being stopped.
        """
        assert stopped_auth_tls_secret.exists, (
            f"TLS Secret '{stopped_auth_tls_secret.name}' no longer exists after upgrade for stopped notebook"
        )
