import pytest
from ocp_resources.cluster_role_binding import ClusterRoleBinding
from ocp_resources.config_map import ConfigMap
from ocp_resources.pod import Pod
from ocp_resources.service import Service

KUBE_RBAC_PROXY_CONTAINER = "kube-rbac-proxy"
KUBE_RBAC_PROXY_PORT = 8443
AUTH_DELEGATOR_ROLE = "system:auth-delegator"


@pytest.mark.usefixtures("capture_notebook_baseline")
class TestPreUpgradeNotebookAuth:
    """Verify kube-rbac-proxy auth resources exist before the platform upgrade.

    Steps:
        1. Verify the notebook pod has a kube-rbac-proxy sidecar container.
        2. Verify the kube-rbac-proxy Service exists.
        3. Verify the kube-rbac-proxy ConfigMap exists.
        4. Verify the auth-delegator ClusterRoleBinding exists.
    """

    @pytest.mark.pre_upgrade
    def test_kube_rbac_proxy_sidecar_present(
        self,
        upgrade_notebook_pod: Pod,
    ) -> None:
        """Given a notebook with inject-auth annotation,
        When the ODH webhook injects the sidecar,
        Then the pod should have a kube-rbac-proxy container.
        """
        containers = upgrade_notebook_pod.instance.spec.containers
        container_names = [container.name for container in containers]

        assert KUBE_RBAC_PROXY_CONTAINER in container_names, (
            f"Pod '{upgrade_notebook_pod.name}' missing '{KUBE_RBAC_PROXY_CONTAINER}' sidecar. "
            f"Containers: {container_names}"
        )

    @pytest.mark.pre_upgrade
    def test_kube_rbac_proxy_service_exists(
        self,
        auth_proxy_service: Service,
    ) -> None:
        """Given a notebook with inject-auth annotation,
        When the ODH controller reconciles,
        Then a kube-rbac-proxy Service should exist with port 8443.
        """
        assert auth_proxy_service.exists, f"kube-rbac-proxy Service '{auth_proxy_service.name}' does not exist"

        port_numbers = [port.port for port in auth_proxy_service.instance.spec.ports]
        assert KUBE_RBAC_PROXY_PORT in port_numbers, (
            f"Service '{auth_proxy_service.name}' missing port {KUBE_RBAC_PROXY_PORT}. Found ports: {port_numbers}"
        )

    @pytest.mark.pre_upgrade
    def test_kube_rbac_proxy_configmap_exists(
        self,
        auth_proxy_configmap: ConfigMap,
    ) -> None:
        """Given a notebook with inject-auth annotation,
        When the ODH controller reconciles,
        Then a kube-rbac-proxy ConfigMap should exist.
        """
        assert auth_proxy_configmap.exists, f"kube-rbac-proxy ConfigMap '{auth_proxy_configmap.name}' does not exist"

    @pytest.mark.pre_upgrade
    def test_auth_delegator_cluster_role_binding_exists(
        self,
        auth_delegator_crb: ClusterRoleBinding,
    ) -> None:
        """Given a notebook with inject-auth annotation,
        When the ODH controller reconciles,
        Then an auth-delegator ClusterRoleBinding should exist referencing system:auth-delegator.
        """
        assert auth_delegator_crb.exists, f"ClusterRoleBinding '{auth_delegator_crb.name}' does not exist"

        role_ref = auth_delegator_crb.instance.roleRef
        assert role_ref.name == AUTH_DELEGATOR_ROLE, (
            f"ClusterRoleBinding '{auth_delegator_crb.name}' has roleRef.name='{role_ref.name}', "
            f"expected '{AUTH_DELEGATOR_ROLE}'"
        )


@pytest.mark.usefixtures("stopped_notebook_pre_upgrade_shutdown")
class TestPreUpgradeStoppedNotebookAuth:
    """Verify kube-rbac-proxy auth resources exist for a stopped notebook before upgrade.

    Steps:
        1. Verify the kube-rbac-proxy Service exists for the stopped notebook.
        2. Verify the kube-rbac-proxy ConfigMap exists for the stopped notebook.
        3. Verify the auth-delegator ClusterRoleBinding exists for the stopped notebook.
    """

    @pytest.mark.pre_upgrade
    def test_stopped_kube_rbac_proxy_service_exists(
        self,
        stopped_auth_proxy_service: Service,
    ) -> None:
        """Given a stopped notebook with inject-auth annotation,
        When the notebook is stopped,
        Then the kube-rbac-proxy Service should still exist with port 8443.
        """
        assert stopped_auth_proxy_service.exists, (
            f"kube-rbac-proxy Service '{stopped_auth_proxy_service.name}' does not exist for stopped notebook"
        )

        port_numbers = [port.port for port in stopped_auth_proxy_service.instance.spec.ports]
        assert KUBE_RBAC_PROXY_PORT in port_numbers, (
            f"Service '{stopped_auth_proxy_service.name}' missing port {KUBE_RBAC_PROXY_PORT}. "
            f"Found ports: {port_numbers}"
        )

    @pytest.mark.pre_upgrade
    def test_stopped_kube_rbac_proxy_configmap_exists(
        self,
        stopped_auth_proxy_configmap: ConfigMap,
    ) -> None:
        """Given a stopped notebook with inject-auth annotation,
        When the notebook is stopped,
        Then the kube-rbac-proxy ConfigMap should still exist.
        """
        assert stopped_auth_proxy_configmap.exists, (
            f"kube-rbac-proxy ConfigMap '{stopped_auth_proxy_configmap.name}' does not exist for stopped notebook"
        )

    @pytest.mark.pre_upgrade
    def test_stopped_auth_delegator_crb_exists(
        self,
        stopped_auth_delegator_crb: ClusterRoleBinding,
    ) -> None:
        """Given a stopped notebook with inject-auth annotation,
        When the notebook is stopped,
        Then the auth-delegator ClusterRoleBinding should still exist referencing system:auth-delegator.
        """
        assert stopped_auth_delegator_crb.exists, (
            f"ClusterRoleBinding '{stopped_auth_delegator_crb.name}' does not exist for stopped notebook"
        )

        role_ref = stopped_auth_delegator_crb.instance.roleRef
        assert role_ref.name == AUTH_DELEGATOR_ROLE, (
            f"ClusterRoleBinding '{stopped_auth_delegator_crb.name}' has roleRef.name='{role_ref.name}', "
            f"expected '{AUTH_DELEGATOR_ROLE}'"
        )


class TestPostUpgradeNotebookAuth:
    """Verify kube-rbac-proxy auth resources survived the platform upgrade.

    Steps:
        1. Verify the sidecar container is still present in the running notebook pod.
        2. Verify the kube-rbac-proxy Service still exists for both running and stopped notebooks.
        3. Verify the kube-rbac-proxy ConfigMap still exists for both running and stopped notebooks.
        4. Verify the auth-delegator ClusterRoleBinding still exists for both running and stopped notebooks.
    """

    @pytest.mark.post_upgrade
    def test_kube_rbac_proxy_sidecar_present_after_upgrade(
        self,
        upgrade_notebook_pod: Pod,
    ) -> None:
        """Given a notebook with auth sidecar existed before upgrade,
        When the upgrade completes,
        Then the pod should still have the kube-rbac-proxy container.
        """
        containers = upgrade_notebook_pod.instance.spec.containers
        container_names = [container.name for container in containers]

        assert KUBE_RBAC_PROXY_CONTAINER in container_names, (
            f"Pod '{upgrade_notebook_pod.name}' lost '{KUBE_RBAC_PROXY_CONTAINER}' sidecar after upgrade. "
            f"Containers: {container_names}"
        )

    @pytest.mark.post_upgrade
    def test_kube_rbac_proxy_service_exists_after_upgrade(
        self,
        auth_proxy_service: Service,
    ) -> None:
        """Given a kube-rbac-proxy Service existed before upgrade,
        When the upgrade completes,
        Then the Service should still exist with port 8443.
        """
        assert auth_proxy_service.exists, (
            f"kube-rbac-proxy Service '{auth_proxy_service.name}' no longer exists after upgrade"
        )

        port_numbers = [port.port for port in auth_proxy_service.instance.spec.ports]
        assert KUBE_RBAC_PROXY_PORT in port_numbers, (
            f"Service '{auth_proxy_service.name}' lost port {KUBE_RBAC_PROXY_PORT} after upgrade. "
            f"Found ports: {port_numbers}"
        )

    @pytest.mark.post_upgrade
    def test_kube_rbac_proxy_configmap_exists_after_upgrade(
        self,
        auth_proxy_configmap: ConfigMap,
    ) -> None:
        """Given a kube-rbac-proxy ConfigMap existed before upgrade,
        When the upgrade completes,
        Then the ConfigMap should still exist.
        """
        assert auth_proxy_configmap.exists, (
            f"kube-rbac-proxy ConfigMap '{auth_proxy_configmap.name}' no longer exists after upgrade"
        )

    @pytest.mark.post_upgrade
    def test_auth_delegator_cluster_role_binding_exists_after_upgrade(
        self,
        auth_delegator_crb: ClusterRoleBinding,
    ) -> None:
        """Given an auth-delegator ClusterRoleBinding existed before upgrade,
        When the upgrade completes,
        Then the ClusterRoleBinding should still exist referencing system:auth-delegator.
        """
        assert auth_delegator_crb.exists, (
            f"ClusterRoleBinding '{auth_delegator_crb.name}' no longer exists after upgrade"
        )

        role_ref = auth_delegator_crb.instance.roleRef
        assert role_ref.name == AUTH_DELEGATOR_ROLE, (
            f"ClusterRoleBinding '{auth_delegator_crb.name}' has roleRef.name='{role_ref.name}' after upgrade, "
            f"expected '{AUTH_DELEGATOR_ROLE}'"
        )

    @pytest.mark.post_upgrade
    def test_stopped_kube_rbac_proxy_service_exists_after_upgrade(
        self,
        stopped_auth_proxy_service: Service,
    ) -> None:
        """Given a stopped notebook's kube-rbac-proxy Service existed before upgrade,
        When the upgrade completes,
        Then the Service should still exist with port 8443 despite the notebook being stopped.
        """
        assert stopped_auth_proxy_service.exists, (
            f"kube-rbac-proxy Service '{stopped_auth_proxy_service.name}' "
            f"no longer exists after upgrade for stopped notebook"
        )

        port_numbers = [port.port for port in stopped_auth_proxy_service.instance.spec.ports]
        assert KUBE_RBAC_PROXY_PORT in port_numbers, (
            f"Service '{stopped_auth_proxy_service.name}' lost port {KUBE_RBAC_PROXY_PORT} after upgrade. "
            f"Found ports: {port_numbers}"
        )

    @pytest.mark.post_upgrade
    def test_stopped_kube_rbac_proxy_configmap_exists_after_upgrade(
        self,
        stopped_auth_proxy_configmap: ConfigMap,
    ) -> None:
        """Given a stopped notebook's kube-rbac-proxy ConfigMap existed before upgrade,
        When the upgrade completes,
        Then the ConfigMap should still exist despite the notebook being stopped.
        """
        assert stopped_auth_proxy_configmap.exists, (
            f"kube-rbac-proxy ConfigMap '{stopped_auth_proxy_configmap.name}' "
            f"no longer exists after upgrade for stopped notebook"
        )

    @pytest.mark.post_upgrade
    def test_stopped_auth_delegator_crb_exists_after_upgrade(
        self,
        stopped_auth_delegator_crb: ClusterRoleBinding,
    ) -> None:
        """Given a stopped notebook's auth-delegator CRB existed before upgrade,
        When the upgrade completes,
        Then the ClusterRoleBinding should still exist referencing system:auth-delegator.
        """
        assert stopped_auth_delegator_crb.exists, (
            f"ClusterRoleBinding '{stopped_auth_delegator_crb.name}' "
            f"no longer exists after upgrade for stopped notebook"
        )

        role_ref = stopped_auth_delegator_crb.instance.roleRef
        assert role_ref.name == AUTH_DELEGATOR_ROLE, (
            f"ClusterRoleBinding '{stopped_auth_delegator_crb.name}' has roleRef.name='{role_ref.name}' "
            f"after upgrade, expected '{AUTH_DELEGATOR_ROLE}'"
        )
