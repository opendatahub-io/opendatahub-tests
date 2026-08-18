"""TC-RBAC: RBAC permission verification for lifecycle signal emission.

Covers RHAISTRAT-1923 — verifies that the EvalHub ServiceAccount and TrustyAI Operator
ServiceAccount have the exact permissions needed to emit Events and patch Jobs, and
that Events are namespace-scoped for tenant isolation.
"""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.role_binding import RoleBinding
from ocp_resources.route import Route
from ocp_resources.service import Service
from pytest_testconfig import config as py_config

from tests.ai_safety.evalhub.k8s_lifecycle_signals.constants import (
    LIFECYCLE_REASON_STARTED,
    LIFECYCLE_SIGNALS_CP_NAMESPACE,
    LIFECYCLE_SIGNALS_CR_NAME,
    LIFECYCLE_SIGNALS_NAMESPACE,
)
from tests.ai_safety.evalhub.k8s_lifecycle_signals.utils import (
    check_rbac_can_i,
    list_events_for_job,
    wait_for_evaluation_job_name,
    wait_for_event,
)
from tests.ai_safety.evalhub.utils import (
    build_evalhub_job_payload,
    submit_evalhub_job,
)

_EVALHUB_SERVER_SA = f"{LIFECYCLE_SIGNALS_CR_NAME}-service"
_OPERATOR_SA = "trustyai-service-operator-controller-manager"


@pytest.mark.ai_safety
class TestRbacVerification:
    """TC-RBAC: ServiceAccount RBAC permission and namespace-scoping verification.

    Verifies that both the EvalHub server and TrustyAI Operator ServiceAccounts have
    the minimum required permissions, and that Events are namespace-scoped.
    """

    @pytest.mark.tier1
    def test_rbac_001_evalhub_sa_can_create_events(
        self,
        admin_client: DynamicClient,
        lifecycle_signals_ready: None,
        lifecycle_signals_namespace: Namespace,
    ) -> None:
        """Given the EvalHub ServiceAccount exists in the control-plane namespace,
        when its permission to create Events is queried in the tenant namespace,
        then SubjectAccessReview reports allowed and a RoleBinding granting this exists."""
        tenant_ns = lifecycle_signals_namespace.name

        can_create = check_rbac_can_i(
            admin_client=admin_client,
            verb="create",
            resource="events",
            sa_namespace=LIFECYCLE_SIGNALS_CP_NAMESPACE,
            sa_name=_EVALHUB_SERVER_SA,
            target_namespace=tenant_ns,
        )

        assert can_create, (
            f"ServiceAccount {_EVALHUB_SERVER_SA} in {LIFECYCLE_SIGNALS_CP_NAMESPACE} "
            f"must have permission to create Events in {tenant_ns}"
        )
        bindings = list(RoleBinding.get(client=admin_client, namespace=tenant_ns))
        evalhub_bindings = [
            b
            for b in bindings
            if any(
                s.get("name") == _EVALHUB_SERVER_SA
                and s.get("kind") == "ServiceAccount"
                and s.get("namespace") == LIFECYCLE_SIGNALS_CP_NAMESPACE
                for s in (b.instance.subjects or [])
            )
        ]
        assert evalhub_bindings, (
            f"No RoleBinding found for SA {_EVALHUB_SERVER_SA} ({LIFECYCLE_SIGNALS_CP_NAMESPACE}) in {tenant_ns}"
        )

    @pytest.mark.tier1
    def test_rbac_002_evalhub_sa_can_patch_jobs(
        self,
        admin_client: DynamicClient,
        lifecycle_signals_ready: None,
        lifecycle_signals_namespace: Namespace,
    ) -> None:
        """Given the EvalHub ServiceAccount exists in the control-plane namespace,
        when its permission to patch batch/v1 Jobs is queried in the tenant namespace,
        then SubjectAccessReview reports allowed."""
        tenant_ns = lifecycle_signals_namespace.name

        can_patch = check_rbac_can_i(
            admin_client=admin_client,
            verb="patch",
            resource="jobs",
            sa_namespace=LIFECYCLE_SIGNALS_CP_NAMESPACE,
            sa_name=_EVALHUB_SERVER_SA,
            target_namespace=tenant_ns,
        )

        assert can_patch, (
            f"ServiceAccount {_EVALHUB_SERVER_SA} in {LIFECYCLE_SIGNALS_CP_NAMESPACE} "
            f"must have permission to patch Jobs in {tenant_ns}"
        )

    @pytest.mark.tier1
    def test_rbac_003_operator_sa_can_create_events(
        self,
        admin_client: DynamicClient,
        lifecycle_signals_ready: None,
    ) -> None:
        """Given the TrustyAI Operator ServiceAccount exists,
        when its permission to create Events is queried in the lifecycle signals namespace,
        then SubjectAccessReview reports allowed."""
        operator_ns = py_config["applications_namespace"]
        can_create = check_rbac_can_i(
            admin_client=admin_client,
            verb="create",
            resource="events",
            sa_namespace=operator_ns,
            sa_name=_OPERATOR_SA,
            target_namespace=LIFECYCLE_SIGNALS_NAMESPACE,
        )

        assert can_create, (
            f"Operator SA {_OPERATOR_SA} in {operator_ns} must have permission "
            f"to create Events in {LIFECYCLE_SIGNALS_NAMESPACE}"
        )

    @pytest.mark.tier1
    def test_rbac_004_events_are_namespace_scoped(
        self,
        admin_client: DynamicClient,
        lifecycle_signals_ready: None,
        lifecycle_signals_route: Route,
        lifecycle_signals_ca_bundle_file: str,
        lifecycle_signals_namespace: Namespace,
        lifecycle_signals_vllm_service: Service,
        lifecycle_signals_tenant_a_rbac: None,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        tenant_b_namespace: Namespace,
    ) -> None:
        """Given an evaluation in tenant-a and a cluster operator who can list Events in both namespaces,
        when tenant-a Events are listed and tenant-b Events are listed,
        then the tenant-a EvaluationStarted Event appears in tenant-a but not in tenant-b."""
        host = lifecycle_signals_route.host
        lifecycle_ns = lifecycle_signals_namespace.name
        tenant_a_ns = tenant_a_namespace.name
        payload = build_evalhub_job_payload(
            model_service_name=lifecycle_signals_vllm_service.name,
            tenant_namespace=lifecycle_ns,
            job_name="tc-rbac-004",
        )
        job_id = submit_evalhub_job(
            host=host,
            token=tenant_a_token,
            ca_bundle_file=lifecycle_signals_ca_bundle_file,
            tenant=tenant_a_ns,
            payload=payload,
        )["resource"]["id"]
        job_name = wait_for_evaluation_job_name(
            admin_client=admin_client,
            namespace=tenant_a_ns,
            evalhub_job_id=job_id,
        )
        wait_for_event(
            admin_client=admin_client,
            job_name=job_name,
            namespace=tenant_a_ns,
            reason=LIFECYCLE_REASON_STARTED,
        )

        lifecycle_events = list_events_for_job(
            admin_client=admin_client,
            job_name=job_name,
            namespace=lifecycle_ns,
            reason=LIFECYCLE_REASON_STARTED,
        )
        tenant_a_events = list_events_for_job(
            admin_client=admin_client,
            job_name=job_name,
            namespace=tenant_a_ns,
            reason=LIFECYCLE_REASON_STARTED,
        )
        tenant_b_events = list_events_for_job(
            admin_client=admin_client,
            job_name=job_name,
            namespace=tenant_b_namespace.name,
            reason=LIFECYCLE_REASON_STARTED,
        )

        assert tenant_a_events, f"Expected EvaluationStarted Event in {tenant_a_ns}, found none"
        assert not tenant_b_events, (
            f"Event for job {job_name} should not appear in {tenant_b_namespace.name}; "
            f"found {len(tenant_b_events)} event(s)"
        )
        assert not lifecycle_events, (
            f"Event for job {job_name} should not appear in {lifecycle_ns}; found {len(lifecycle_events)} event(s)"
        )
