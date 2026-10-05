"""Kubernetes resource assertions for EvalHub single-tenancy mode."""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.route import Route
from ocp_resources.service import Service
from ocp_resources.service_account import ServiceAccount
from ocp_resources.service_monitor import ServiceMonitor

from tests.ai_safety.evalhub.constants import EVALHUB_METRICS_SERVICE_SUFFIX, EVALHUB_PLURAL
from tests.ai_safety.evalhub.single_tenancy.constants import (
    EVALHUB_DISCOVERY_CM_NAME,
    EVALHUB_ST_CR_NAME,
    EVALHUB_TENANT_ADMIN_BINDING_NAME,
    EVALHUB_TENANT_ADMIN_ROLE_NAME,
    EVALHUB_TRUSTYAI_API_GROUP,
    EVALHUB_USER_ROLE_NAME,
)
from tests.ai_safety.evalhub.single_tenancy.utils import SingleTenantEvalHub


@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-evalhub-st-resources"})],
    indirect=True,
)
@pytest.mark.smoke
@pytest.mark.ai_safety
class TestEvalHubSingleTenancyOperatorResources:
    """Verifies the operator creates the expected Deployment, Service, Route,
    metrics Service, and ServiceMonitor when spec.tenancy: single.
    """

    def test_deployment_created_and_ready(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_st_deployment: Deployment,
    ) -> None:
        """Given: an EvalHub CR with spec.tenancy: single in a workload namespace.

        When: the operator reconciles the instance.

        Then: a Deployment named after the CR exists and has at least one ready replica.
        """
        assert evalhub_st_deployment.exists, f"Expected Deployment '{EVALHUB_ST_CR_NAME}' in {model_namespace.name}"
        status = evalhub_st_deployment.instance.status
        assert status.readyReplicas and status.readyReplicas >= 1, (
            f"Deployment not ready: readyReplicas={status.readyReplicas}"
        )

    def test_service_created_with_correct_port(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_st_cr: SingleTenantEvalHub,
    ) -> None:
        """Given: a single-tenancy EvalHub instance in a workload namespace.

        When: the operator-created Service is inspected.

        Then: it exists on port 8443 with selector instance matching the CR name.
        """
        svc = Service(
            client=admin_client,
            name=EVALHUB_ST_CR_NAME,
            namespace=model_namespace.name,
        )
        assert svc.exists, f"Expected Service '{EVALHUB_ST_CR_NAME}' in {model_namespace.name}"

        ports = svc.instance.spec.ports
        port_numbers = [port.port for port in ports]
        assert 8443 in port_numbers, f"Expected port 8443, found: {port_numbers}"

        selector = dict(svc.instance.spec.selector or {})
        assert selector.get("instance") == EVALHUB_ST_CR_NAME, (
            f"Expected selector instance={EVALHUB_ST_CR_NAME}, got: {selector}"
        )

    def test_route_created(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_st_route: Route,
    ) -> None:
        """Given: a single-tenancy EvalHub instance in a workload namespace.

        When: the operator-created Route is inspected.

        Then: the Route exists and has a host assigned by the OpenShift router.
        """
        assert evalhub_st_route.exists, f"Expected Route '{EVALHUB_ST_CR_NAME}' in {model_namespace.name}"
        host = evalhub_st_route.host
        assert host, "Route has no host — OpenShift router has not assigned one yet"

    def test_metrics_service_created(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_st_cr: SingleTenantEvalHub,
    ) -> None:
        """Given: a single-tenancy EvalHub instance in a workload namespace.

        When: the operator-created metrics Service is inspected.

        Then: it exists and exposes port 8081 for Prometheus scraping.
        """
        metrics_name = f"{EVALHUB_ST_CR_NAME}{EVALHUB_METRICS_SERVICE_SUFFIX}"
        svc = Service(
            client=admin_client,
            name=metrics_name,
            namespace=model_namespace.name,
        )
        assert svc.exists, f"Expected metrics Service '{metrics_name}' in {model_namespace.name}"

        ports = svc.instance.spec.ports
        port_numbers = [port.port for port in ports]
        assert 8081 in port_numbers, f"Expected metrics port 8081, found: {port_numbers}"

    def test_service_monitor_created(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_st_cr: SingleTenantEvalHub,
    ) -> None:
        """Given: a single-tenancy EvalHub instance in a workload namespace.

        When: the operator-created ServiceMonitor is inspected.

        Then: it exists and targets the metrics Service.
        """
        sm_name = f"{EVALHUB_ST_CR_NAME}{EVALHUB_METRICS_SERVICE_SUFFIX}"
        sm = ServiceMonitor(
            client=admin_client,
            name=sm_name,
            namespace=model_namespace.name,
        )
        assert sm.exists, f"Expected ServiceMonitor '{sm_name}' in {model_namespace.name}"


@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-evalhub-st-rbac"})],
    indirect=True,
)
@pytest.mark.tier1
@pytest.mark.ai_safety
class TestEvalHubSingleTenancyRBAC:
    """Verifies the three convenience RBAC objects (evalhub-tenant-admin Role,
    evalhub-user Role, evalhub-tenant-admin-binding RoleBinding) that the
    operator creates exclusively in single-tenancy mode, and that each is
    owner-ref'd to the EvalHub CR for automatic GC on deletion.

    These three objects are exclusive to single-tenancy — the operator deletes them
    when the CR switches to multi mode and re-creates them on switch back.
    """

    def test_tenant_admin_role_created(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_st_cr: SingleTenantEvalHub,
        evalhub_st_deployment: Deployment,
    ) -> None:
        """Given: a single-tenancy EvalHub instance with a ready Deployment.

        When: the evalhub-tenant-admin Role is inspected.

        Then: it exists with rules granting access to trustyai.opendatahub.io resources.
        """
        role = Role(
            client=admin_client,
            name=EVALHUB_TENANT_ADMIN_ROLE_NAME,
            namespace=model_namespace.name,
        )
        assert role.exists, f"Expected Role '{EVALHUB_TENANT_ADMIN_ROLE_NAME}' in {model_namespace.name}"
        rules = role.instance.rules or []
        assert rules, "evalhub-tenant-admin Role has no rules"
        api_groups = [api_group for rule in rules for api_group in (rule.apiGroups or [])]
        assert EVALHUB_TRUSTYAI_API_GROUP in api_groups, (
            f"Expected apiGroup '{EVALHUB_TRUSTYAI_API_GROUP}' in admin Role, got: {api_groups}"
        )

    def test_user_role_created(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_st_cr: SingleTenantEvalHub,
        evalhub_st_deployment: Deployment,
    ) -> None:
        """Given: a single-tenancy EvalHub instance with a ready Deployment.

        When: the evalhub-user Role is inspected.

        Then: it exists with at least one RBAC rule.
        """
        role = Role(
            client=admin_client,
            name=EVALHUB_USER_ROLE_NAME,
            namespace=model_namespace.name,
        )
        assert role.exists, f"Expected Role '{EVALHUB_USER_ROLE_NAME}' in {model_namespace.name}"
        rules = role.instance.rules or []
        assert rules, "evalhub-user Role has no rules"

    def test_tenant_admin_binding_created(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_st_cr: SingleTenantEvalHub,
        evalhub_st_deployment: Deployment,
    ) -> None:
        """Given: a single-tenancy EvalHub instance with a ready Deployment.

        When: the evalhub-tenant-admin-binding RoleBinding is inspected.

        Then: it exists in the workload namespace.
        """
        rb = RoleBinding(
            client=admin_client,
            name=EVALHUB_TENANT_ADMIN_BINDING_NAME,
            namespace=model_namespace.name,
        )
        assert rb.exists, f"Expected RoleBinding '{EVALHUB_TENANT_ADMIN_BINDING_NAME}' in {model_namespace.name}"

    def test_binding_references_admin_role(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_st_cr: SingleTenantEvalHub,
    ) -> None:
        """Given: a single-tenancy EvalHub instance in a workload namespace.

        When: the evalhub-tenant-admin-binding RoleBinding roleRef is inspected.

        Then: roleRef kind is Role and name is evalhub-tenant-admin.
        """
        rb = RoleBinding(
            client=admin_client,
            name=EVALHUB_TENANT_ADMIN_BINDING_NAME,
            namespace=model_namespace.name,
            ensure_exists=True,
        )
        role_ref = rb.instance.roleRef
        assert role_ref.kind == "Role", f"Expected roleRef.kind 'Role', got '{role_ref.kind}'"
        assert role_ref.name == EVALHUB_TENANT_ADMIN_ROLE_NAME, (
            f"Expected roleRef.name '{EVALHUB_TENANT_ADMIN_ROLE_NAME}', got '{role_ref.name}'"
        )

    def test_binding_subject_is_all_namespace_sas(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_st_cr: SingleTenantEvalHub,
    ) -> None:
        """Given: a single-tenancy EvalHub instance in a workload namespace.

        When: the evalhub-tenant-admin-binding subjects are inspected.

        Then: a Group subject named system:serviceaccounts:{namespace} is present.
        """
        rb = RoleBinding(
            client=admin_client,
            name=EVALHUB_TENANT_ADMIN_BINDING_NAME,
            namespace=model_namespace.name,
            ensure_exists=True,
        )
        subjects = rb.instance.subjects or []
        assert subjects, "evalhub-tenant-admin-binding has no subjects"
        expected_name = f"system:serviceaccounts:{model_namespace.name}"
        subject_names = [subject.name for subject in subjects]
        assert expected_name in subject_names, f"Expected subject '{expected_name}' in binding, got: {subject_names}"
        ns_group = next(subject for subject in subjects if subject.name == expected_name)
        assert ns_group.kind == "Group", f"Expected subject kind 'Group', got '{ns_group.kind}'"

    def test_admin_role_has_owner_reference(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_st_cr: SingleTenantEvalHub,
    ) -> None:
        """Given: a single-tenancy EvalHub instance in a workload namespace.

        When: the evalhub-tenant-admin Role ownerReferences are inspected.

        Then: an EvalHub ownerReference pointing to the CR name is present.
        """
        role = Role(
            client=admin_client,
            name=EVALHUB_TENANT_ADMIN_ROLE_NAME,
            namespace=model_namespace.name,
            ensure_exists=True,
        )
        refs = role.instance.metadata.ownerReferences or []
        assert refs, f"Role '{EVALHUB_TENANT_ADMIN_ROLE_NAME}' has no ownerReferences"
        evalhub_ref = next((owner_ref for owner_ref in refs if owner_ref.kind == "EvalHub"), None)
        assert evalhub_ref is not None, f"No EvalHub ownerReference on Role '{EVALHUB_TENANT_ADMIN_ROLE_NAME}': {refs}"
        assert evalhub_ref.name == evalhub_st_cr.name, (
            f"ownerReference.name '{evalhub_ref.name}' != CR name '{evalhub_st_cr.name}'"
        )

    def test_user_role_has_owner_reference(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_st_cr: SingleTenantEvalHub,
    ) -> None:
        """Given: a single-tenancy EvalHub instance in a workload namespace.

        When: the evalhub-user Role ownerReferences are inspected.

        Then: an EvalHub ownerReference pointing to the CR name is present.
        """
        role = Role(
            client=admin_client,
            name=EVALHUB_USER_ROLE_NAME,
            namespace=model_namespace.name,
            ensure_exists=True,
        )
        refs = role.instance.metadata.ownerReferences or []
        assert refs, f"Role '{EVALHUB_USER_ROLE_NAME}' has no ownerReferences"
        evalhub_ref = next((owner_ref for owner_ref in refs if owner_ref.kind == "EvalHub"), None)
        assert evalhub_ref is not None, f"No EvalHub ownerReference on Role '{EVALHUB_USER_ROLE_NAME}': {refs}"
        assert evalhub_ref.name == evalhub_st_cr.name

    def test_binding_has_owner_reference(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_st_cr: SingleTenantEvalHub,
    ) -> None:
        """Given: a single-tenancy EvalHub instance in a workload namespace.

        When: the evalhub-tenant-admin-binding ownerReferences are inspected.

        Then: an EvalHub ownerReference pointing to the CR name is present.
        """
        rb = RoleBinding(
            client=admin_client,
            name=EVALHUB_TENANT_ADMIN_BINDING_NAME,
            namespace=model_namespace.name,
            ensure_exists=True,
        )
        refs = rb.instance.metadata.ownerReferences or []
        assert refs, f"RoleBinding '{EVALHUB_TENANT_ADMIN_BINDING_NAME}' has no ownerReferences"
        evalhub_ref = next((owner_ref for owner_ref in refs if owner_ref.kind == "EvalHub"), None)
        assert evalhub_ref is not None, (
            f"No EvalHub ownerReference on RoleBinding '{EVALHUB_TENANT_ADMIN_BINDING_NAME}': {refs}"
        )
        assert evalhub_ref.name == evalhub_st_cr.name

    def test_tenant_admin_role_grants_evalhubs_read(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_st_cr: SingleTenantEvalHub,
        evalhub_st_deployment: Deployment,
    ) -> None:
        """Given: a single-tenancy EvalHub instance in a workload namespace.

        When: the policy rules of the evalhub-tenant-admin Role are inspected.

        Then: a rule granting get and list on evalhubs exists, so the BFF can
        look up Status.URL directly from the EvalHub CR without a discovery ConfigMap.
        """
        role = Role(
            client=admin_client,
            name=EVALHUB_TENANT_ADMIN_ROLE_NAME,
            namespace=model_namespace.name,
            ensure_exists=True,
        )
        evalhubs_rule = next(
            (
                rule
                for rule in (role.instance.rules or [])
                if EVALHUB_TRUSTYAI_API_GROUP in (rule.apiGroups or []) and EVALHUB_PLURAL in (rule.resources or [])
            ),
            None,
        )
        assert evalhubs_rule is not None, (
            f"Role '{EVALHUB_TENANT_ADMIN_ROLE_NAME}' has no rule for '{EVALHUB_TRUSTYAI_API_GROUP}/{EVALHUB_PLURAL}'"
        )
        verbs = list(evalhubs_rule.verbs or [])
        assert "get" in verbs, f"Expected 'get' in evalhubs rule verbs, got: {verbs}"
        assert "list" in verbs, f"Expected 'list' in evalhubs rule verbs, got: {verbs}"

    def test_tenant_user_role_grants_evalhubs_read(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_st_cr: SingleTenantEvalHub,
        evalhub_st_deployment: Deployment,
    ) -> None:
        """Given: a single-tenancy EvalHub instance in a workload namespace.

        When: the policy rules of the evalhub-user Role are inspected.

        Then: a rule granting get and list on evalhubs exists, so the BFF can
        look up Status.URL directly from the EvalHub CR without a discovery ConfigMap.
        """
        role = Role(
            client=admin_client,
            name=EVALHUB_USER_ROLE_NAME,
            namespace=model_namespace.name,
            ensure_exists=True,
        )
        evalhubs_rule = next(
            (
                rule
                for rule in (role.instance.rules or [])
                if EVALHUB_TRUSTYAI_API_GROUP in (rule.apiGroups or []) and EVALHUB_PLURAL in (rule.resources or [])
            ),
            None,
        )
        assert evalhubs_rule is not None, (
            f"Role '{EVALHUB_USER_ROLE_NAME}' has no rule for '{EVALHUB_TRUSTYAI_API_GROUP}/{EVALHUB_PLURAL}'"
        )
        verbs = list(evalhubs_rule.verbs or [])
        assert "get" in verbs, f"Expected 'get' in evalhubs rule verbs, got: {verbs}"
        assert "list" in verbs, f"Expected 'list' in evalhubs rule verbs, got: {verbs}"


@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-evalhub-st-isolation"})],
    indirect=True,
)
@pytest.mark.tier1
@pytest.mark.ai_safety
class TestEvalHubSingleTenancyNoCrossNamespace:
    """Verifies that a single-tenant EvalHub does not provision any resources
    (discovery ConfigMap, job ServiceAccount, RoleBindings) in a second
    unlabeled namespace.

    Unlike multi-tenancy, no discovery ConfigMaps, job ServiceAccounts, or RoleBindings
    should appear in namespaces other than the one where the EvalHub CR lives.
    """

    def test_no_discovery_configmap_in_other_namespace(
        self,
        admin_client: DynamicClient,
        second_namespace: Namespace,
        evalhub_st_cr: SingleTenantEvalHub,
    ) -> None:
        """Given: a single-tenancy EvalHub in one namespace and an unlabeled second namespace.

        When: the second namespace is checked for an evalhub-discovery ConfigMap.

        Then: no such ConfigMap exists.
        """
        cm = ConfigMap(
            client=admin_client,
            name=EVALHUB_DISCOVERY_CM_NAME,
            namespace=second_namespace.name,
        )
        assert not cm.exists, (
            f"Unexpected ConfigMap '{EVALHUB_DISCOVERY_CM_NAME}' found in {second_namespace.name}: "
            "single-tenant EvalHub must not provision discovery CMs in other namespaces"
        )

    def test_no_job_service_account_in_other_namespace(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        second_namespace: Namespace,
        evalhub_st_cr: SingleTenantEvalHub,
    ) -> None:
        """Given: a single-tenancy EvalHub in one namespace and an unlabeled second namespace.

        When: the second namespace is checked for the EvalHub job ServiceAccount.

        Then: no job ServiceAccount exists there.
        """
        expected_sa_name = f"{evalhub_st_cr.name}-{model_namespace.name}-job"
        sa = ServiceAccount(
            client=admin_client,
            name=expected_sa_name,
            namespace=second_namespace.name,
        )
        assert not sa.exists, (
            f"Unexpected EvalHub job ServiceAccount '{expected_sa_name}' in {second_namespace.name}: "
            "single-tenant EvalHub must not provision job ServiceAccounts in other namespaces"
        )

    def test_no_rolebindings_in_other_namespace(
        self,
        admin_client: DynamicClient,
        second_namespace: Namespace,
        evalhub_st_cr: SingleTenantEvalHub,
    ) -> None:
        """Given: a single-tenancy EvalHub in one namespace and an unlabeled second namespace.

        When: RoleBindings in the second namespace are listed.

        Then: none have names prefixed with the EvalHub CR name.
        """
        rbs = list(RoleBinding.get(client=admin_client, namespace=second_namespace.name))
        cr_name = evalhub_st_cr.name
        evalhub_rbs = [rb for rb in rbs if rb.name.startswith(cr_name)]
        assert not evalhub_rbs, (
            f"Unexpected EvalHub RoleBinding(s) in {second_namespace.name}: {[rb.name for rb in evalhub_rbs]}"
        )
