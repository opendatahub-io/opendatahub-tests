import pytest
from ocp_resources.cluster_role_binding import ClusterRoleBinding
from ocp_resources.config_map import ConfigMap
from ocp_resources.notebook import Notebook
from ocp_resources.pod import Pod
from ocp_resources.service import Service

from tests.workbenches.notebooks_server.controller.upgrade.test_upgrade_routing import (
    GATEWAY_NAME,
    GATEWAY_NAMESPACE,
    _assert_notebook_backend,
)
from tests.workbenches.notebooks_server.controller.utils import MutatingWebhookConfiguration, StatefulSet
from utilities.resources.http_route import HTTPRoute

KUBE_RBAC_PROXY_CONTAINER = "kube-rbac-proxy"
KUBE_RBAC_PROXY_PORT = 8443
AUTH_DELEGATOR_ROLE = "system:auth-delegator"
EXPECTED_WEBHOOK_FAILURE_POLICY = "Fail"


class TestPostUpgradeNotebookCreation:
    """Verify a new notebook can be created on the upgraded platform.

    Steps:
        1. Create a fresh Notebook CR on the upgraded controller.
        2. Verify the pod reaches Ready state.
        3. Verify the StatefulSet and Service are created.
        4. Verify the HTTPRoute is created with correct gateway parent and backend refs.
        5. Verify the kube-rbac-proxy sidecar is injected.
        6. Verify auth resources (Service, ConfigMap, ClusterRoleBinding) are reconciled.
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
    def test_new_notebook_httproute_exists(
        self,
        new_notebook: Notebook,
        new_notebook_httproute: HTTPRoute,
    ) -> None:
        """Given a new notebook is created post-upgrade,
        When the ODH controller reconciles routing,
        Then an HTTPRoute should be created with correct gateway parent and backend refs.
        """
        assert new_notebook_httproute.exists, (
            f"HTTPRoute '{new_notebook_httproute.name}' was not created on upgraded platform"
        )

        parent_refs = new_notebook_httproute.instance.spec.get("parentRefs", [])
        has_gateway = any(
            ref.get("name") == GATEWAY_NAME and ref.get("namespace") == GATEWAY_NAMESPACE for ref in parent_refs
        )
        assert has_gateway, (
            f"HTTPRoute '{new_notebook_httproute.name}' does not reference "
            f"'{GATEWAY_NAMESPACE}/{GATEWAY_NAME}'. parentRefs: {parent_refs}"
        )

        _assert_notebook_backend(route=new_notebook_httproute, notebook=new_notebook)

    @pytest.mark.post_upgrade
    def test_new_notebook_has_auth_sidecar(
        self,
        new_notebook_pod: Pod,
    ) -> None:
        """Given a new notebook is created post-upgrade with inject-auth,
        When the ODH webhook mutates the CR,
        Then the pod should have a kube-rbac-proxy sidecar container.
        """
        containers = new_notebook_pod.instance.spec.containers
        container_names = [container.name for container in containers]

        assert KUBE_RBAC_PROXY_CONTAINER in container_names, (
            f"Pod '{new_notebook_pod.name}' missing '{KUBE_RBAC_PROXY_CONTAINER}' sidecar on upgraded platform. "
            f"Containers: {container_names}"
        )

    @pytest.mark.post_upgrade
    def test_new_notebook_auth_proxy_service_exists(
        self,
        new_notebook_auth_proxy_service: Service,
    ) -> None:
        """Given a new notebook is created post-upgrade with inject-auth,
        When the controller reconciles auth resources,
        Then a kube-rbac-proxy Service should be created with port 8443.
        """
        assert new_notebook_auth_proxy_service.exists, (
            f"kube-rbac-proxy Service '{new_notebook_auth_proxy_service.name}' was not created on upgraded platform"
        )

        port_numbers = [port.port for port in new_notebook_auth_proxy_service.instance.spec.ports]
        assert KUBE_RBAC_PROXY_PORT in port_numbers, (
            f"Service '{new_notebook_auth_proxy_service.name}' missing port {KUBE_RBAC_PROXY_PORT}. "
            f"Found ports: {port_numbers}"
        )

    @pytest.mark.post_upgrade
    def test_new_notebook_auth_proxy_configmap_exists(
        self,
        new_notebook_auth_proxy_configmap: ConfigMap,
    ) -> None:
        """Given a new notebook is created post-upgrade with inject-auth,
        When the controller reconciles auth resources,
        Then a kube-rbac-proxy ConfigMap should be created.
        """
        assert new_notebook_auth_proxy_configmap.exists, (
            f"kube-rbac-proxy ConfigMap '{new_notebook_auth_proxy_configmap.name}' was not created on upgraded platform"
        )

    @pytest.mark.post_upgrade
    def test_new_notebook_auth_delegator_crb_exists(
        self,
        new_notebook_auth_delegator_crb: ClusterRoleBinding,
    ) -> None:
        """Given a new notebook is created post-upgrade with inject-auth,
        When the controller reconciles auth resources,
        Then an auth-delegator ClusterRoleBinding should be created referencing system:auth-delegator.
        """
        assert new_notebook_auth_delegator_crb.exists, (
            f"ClusterRoleBinding '{new_notebook_auth_delegator_crb.name}' was not created on upgraded platform"
        )

        role_ref = new_notebook_auth_delegator_crb.instance.roleRef
        assert role_ref.name == AUTH_DELEGATOR_ROLE, (
            f"ClusterRoleBinding '{new_notebook_auth_delegator_crb.name}' has roleRef.name='{role_ref.name}', "
            f"expected '{AUTH_DELEGATOR_ROLE}'"
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


class TestPostUpgradeWebhookAvailability:
    """Verify the notebook mutating webhook is available after upgrade.

    Steps:
        1. Confirm the MutatingWebhookConfiguration resource exists.
        2. Verify failurePolicy is 'Fail' (not degraded to 'Ignore').
        3. Verify the webhook targets the correct resource (notebooks).
    """

    @pytest.mark.post_upgrade
    def test_mutating_webhook_exists(
        self,
        notebook_mutating_webhook: MutatingWebhookConfiguration,
    ) -> None:
        """Given the platform was upgraded,
        When the ODH notebook controller is deployed,
        Then the MutatingWebhookConfiguration should exist.
        """
        assert notebook_mutating_webhook.exists, (
            f"MutatingWebhookConfiguration '{notebook_mutating_webhook.name}' not found after upgrade"
        )

    @pytest.mark.post_upgrade
    def test_mutating_webhook_failure_policy(
        self,
        notebook_mutating_webhook: MutatingWebhookConfiguration,
    ) -> None:
        """Given the webhook exists post-upgrade,
        When checking its configuration,
        Then failurePolicy should be 'Fail' to prevent unvalidated notebooks.
        """
        assert notebook_mutating_webhook.exists, (
            f"MutatingWebhookConfiguration '{notebook_mutating_webhook.name}' not found after upgrade"
        )
        webhooks = notebook_mutating_webhook.instance.webhooks
        assert webhooks, f"MutatingWebhookConfiguration '{notebook_mutating_webhook.name}' has no webhooks defined"

        for webhook in webhooks:
            failure_policy = webhook.failurePolicy
            assert failure_policy == EXPECTED_WEBHOOK_FAILURE_POLICY, (
                f"Webhook '{webhook.name}' has failurePolicy='{failure_policy}', "
                f"expected '{EXPECTED_WEBHOOK_FAILURE_POLICY}'"
            )

    @pytest.mark.post_upgrade
    def test_mutating_webhook_targets_notebooks(
        self,
        notebook_mutating_webhook: MutatingWebhookConfiguration,
    ) -> None:
        """Given the webhook exists post-upgrade,
        When checking its rules,
        Then it should target kubeflow.org notebooks for CREATE and UPDATE.
        """
        assert notebook_mutating_webhook.exists, (
            f"MutatingWebhookConfiguration '{notebook_mutating_webhook.name}' not found after upgrade"
        )
        webhooks = notebook_mutating_webhook.instance.webhooks
        assert webhooks, f"MutatingWebhookConfiguration '{notebook_mutating_webhook.name}' has no webhooks defined"

        notebook_rules_found = False
        notebook_ops: set[str] = set()
        for webhook in webhooks:
            for rule in webhook.rules or []:
                api_groups = list(rule.apiGroups or [])
                resources = list(rule.resources or [])
                operations = list(rule.operations or [])
                if "kubeflow.org" in api_groups and "notebooks" in resources:
                    notebook_rules_found = True
                    notebook_ops.update(operations)

        assert notebook_rules_found, (
            f"No webhook rule found targeting 'kubeflow.org/notebooks' in '{notebook_mutating_webhook.name}'"
        )
        assert "CREATE" in notebook_ops, (
            f"No notebook webhook rule includes CREATE operation in '{notebook_mutating_webhook.name}'"
        )
        assert "UPDATE" in notebook_ops, (
            f"No notebook webhook rule includes UPDATE operation in '{notebook_mutating_webhook.name}'"
        )
