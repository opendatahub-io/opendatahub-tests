"""Lifecycle tests for EvalHub single-tenancy mode.

Five test classes covering state transitions and edge cases:

  TestEvalHubInvalidPlacement
      A multi-tenant EvalHub CR placed in a namespace already labelled as
      a tenant must transition to Error/InvalidPlacement phase and must
      not create a Deployment.

  TestEvalHubModeSwitchSingleToMulti
      An EvalHub CR starting in single mode (with convenience Roles) is
      patched to multi. Verifies the Roles are removed and cross-namespace
      RBAC is provisioned in the labeled tenant namespace.

  TestEvalHubModeSwitchMultiToSingle
      An EvalHub CR starting in multi mode is patched to single. Verifies
      convenience Roles are created and cross-namespace RoleBindings are
      cleaned up from the tenant namespace.

  TestEvalHubTenantConfigMapHotMount
      A ConfigMap labeled with the provider-type=tenant label triggers the
      operator's ConfigMap watch, causing the Deployment generation to
      increment (new rollout). Verifies the CM is referenced in Deployment volumes.

  TestEvalHubSingleTenancyDeletion
      Deleting a single-tenant EvalHub CR causes the owner-referenced Roles
      and RoleBinding to be garbage-collected.

Run in isolation:
    pytest tests/ai_safety/evalhub/single_tenancy/test_evalhub_single_tenancy_lifecycle.py -m ai_safety
"""

from collections.abc import Generator
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.evalhub import EvalHub
from ocp_resources.namespace import Namespace
from ocp_resources.resource import ResourceEditor
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from timeout_sampler import TimeoutSampler

from tests.ai_safety.evalhub.constants import (
    EVALHUB_JOB_CONFIG_CLUSTERROLE,
    EVALHUB_JOBS_WRITER_CLUSTERROLE,
    EVALHUB_TENANT_LABEL_KEY,
    EVALHUB_TENANT_LABEL_VALUE,
)
from tests.ai_safety.evalhub.single_tenancy.constants import (
    EVALHUB_TENANCY_MULTI,
    EVALHUB_TENANCY_SINGLE,
    EVALHUB_TENANT_ADMIN_BINDING_NAME,
    EVALHUB_TENANT_ADMIN_ROLE_NAME,
    EVALHUB_TENANT_PROVIDER_LABEL_KEY,
    EVALHUB_TENANT_PROVIDER_LABEL_VAL,
    EVALHUB_USER_ROLE_NAME,
)
from tests.ai_safety.evalhub.single_tenancy.utils import SingleTenantEvalHub

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-evalhub-st-invalid-placement"})],
    indirect=True,
)
@pytest.mark.tier2
@pytest.mark.ai_safety
class TestEvalHubInvalidPlacement:
    """A multi-tenant EvalHub placed in a tenant-labelled namespace enters Error/InvalidPlacement.

    The operator detects the conflict on the third reconcile (after status init and finalizer
    addition) and calls DoNotRequeue() — no Deployment is ever created.
    """

    @pytest.fixture(scope="class")
    def invalid_placement_namespace(
        self,
        model_namespace: Namespace,
    ) -> Namespace:
        """Re-label the model_namespace with the tenant label before EvalHub is created."""
        ResourceEditor(
            patches={
                model_namespace: {
                    "metadata": {"labels": {EVALHUB_TENANT_LABEL_KEY: EVALHUB_TENANT_LABEL_VALUE}},
                },
            },
        ).update()
        return model_namespace

    @pytest.fixture(scope="class")
    def invalid_placement_evalhub(
        self,
        admin_client: DynamicClient,
        invalid_placement_namespace: Namespace,
    ) -> Generator[EvalHub, Any, Any]:
        """Multi-tenant (default) EvalHub CR in the tenant-labelled namespace."""
        with EvalHub(
            client=admin_client,
            name="evalhub-invalid",
            namespace=invalid_placement_namespace.name,
            database={"type": "sqlite"},
            wait_for_resource=True,
        ) as evalhub:
            yield evalhub

    def test_invalid_placement_sets_error_phase(
        self,
        invalid_placement_evalhub: EvalHub,
    ) -> None:
        """Given: a multi-tenant (default) EvalHub CR in a namespace labelled as a tenant.

        When: the operator reconciles and detects the placement conflict.

        Then: status.phase transitions to Error.
        """

        def _get_phase() -> str | None:
            return (invalid_placement_evalhub.instance.status or {}).get("phase")

        for phase in TimeoutSampler(
            wait_timeout=120,
            sleep=5,
            func=_get_phase,
        ):
            if phase == "Error":
                LOGGER.info(f"InvalidPlacement error phase confirmed for '{invalid_placement_evalhub.name}'")
                return

    def test_invalid_placement_no_deployment(
        self,
        admin_client: DynamicClient,
        invalid_placement_namespace: Namespace,
        invalid_placement_evalhub: EvalHub,
    ) -> None:
        """Given: a multi-tenant EvalHub CR in InvalidPlacement (tenant-labelled namespace).

        When: the operator halts reconciliation without creating a Deployment.

        Then: no Deployment named after the CR exists in the namespace.
        """
        deployment = Deployment(
            client=admin_client,
            name=invalid_placement_evalhub.name,
            namespace=invalid_placement_namespace.name,
        )
        assert not deployment.exists, (
            f"Unexpected Deployment '{invalid_placement_evalhub.name}' found: "
            "operator must not create a Deployment for an InvalidPlacement CR"
        )


@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-evalhub-st-switch-to-mt"})],
    indirect=True,
)
@pytest.mark.tier2
@pytest.mark.ai_safety
class TestEvalHubModeSwitchSingleToMulti:
    """EvalHub starts in single mode; patching spec.tenancy to multi removes Roles
    and provisions RBAC in newly labeled tenant namespaces.
    """

    def test_roles_exist_in_single_mode(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_st_cr: SingleTenantEvalHub,
        evalhub_st_deployment: Deployment,
    ) -> None:
        """Given: an EvalHub CR with spec.tenancy: single and a ready Deployment.

        When: the convenience Roles are inspected before any mode switch.

        Then: evalhub-tenant-admin and evalhub-user Roles exist in the workload namespace.
        """
        admin_role = Role(
            client=admin_client,
            name=EVALHUB_TENANT_ADMIN_ROLE_NAME,
            namespace=model_namespace.name,
        )
        user_role = Role(
            client=admin_client,
            name=EVALHUB_USER_ROLE_NAME,
            namespace=model_namespace.name,
        )
        assert admin_role.exists, (
            f"Pre-condition failed: Role '{EVALHUB_TENANT_ADMIN_ROLE_NAME}' not found "
            f"in {model_namespace.name} before mode switch"
        )
        assert user_role.exists, (
            f"Pre-condition failed: Role '{EVALHUB_USER_ROLE_NAME}' not found "
            f"in {model_namespace.name} before mode switch"
        )

    def test_switch_to_multi_removes_roles(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_st_cr: SingleTenantEvalHub,
    ) -> None:
        """Given: a single-tenancy EvalHub with convenience Roles in its namespace.

        When: spec.tenancy is patched to multi.

        Then: evalhub-tenant-admin, evalhub-user, and evalhub-tenant-admin-binding are deleted.
        """
        ResourceEditor(patches={evalhub_st_cr: {"spec": {"tenancy": EVALHUB_TENANCY_MULTI}}}).update()
        LOGGER.info(f"Patched '{evalhub_st_cr.name}' spec.tenancy → multi")

        def _roles_gone() -> bool:
            admin_role = Role(client=admin_client, name=EVALHUB_TENANT_ADMIN_ROLE_NAME, namespace=model_namespace.name)
            user_role = Role(client=admin_client, name=EVALHUB_USER_ROLE_NAME, namespace=model_namespace.name)
            binding = RoleBinding(
                client=admin_client, name=EVALHUB_TENANT_ADMIN_BINDING_NAME, namespace=model_namespace.name
            )
            return not admin_role.exists and not user_role.exists and not binding.exists

        for gone in TimeoutSampler(wait_timeout=120, sleep=5, func=_roles_gone):
            if gone:
                LOGGER.info("Convenience Roles removed after switch to multi mode")
                return

    def test_switch_to_multi_provisions_tenant_namespace(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        labeled_tenant_namespace: Namespace,
        evalhub_st_cr: SingleTenantEvalHub,
    ) -> None:
        """Given: a single-tenancy EvalHub switched to multi mode with a labeled tenant namespace.

        When: the operator reconciles cross-namespace tenant RBAC.

        Then: jobs-writer and job-config RoleBindings are created in the tenant namespace.
        """
        cr_name = evalhub_st_cr.name

        def _tenant_rbac_ready() -> bool:
            rbs = list(RoleBinding.get(client=admin_client, namespace=labeled_tenant_namespace.name))
            has_job_writer = any(
                rb.instance.roleRef.name == EVALHUB_JOBS_WRITER_CLUSTERROLE and rb.name.startswith(cr_name)
                for rb in rbs
            )
            has_job_config = any(
                rb.instance.roleRef.name == EVALHUB_JOB_CONFIG_CLUSTERROLE and rb.name.startswith(cr_name) for rb in rbs
            )
            return has_job_writer and has_job_config

        for ready in TimeoutSampler(wait_timeout=180, sleep=5, func=_tenant_rbac_ready):
            if ready:
                LOGGER.info(f"Multi-tenant RBAC provisioned in {labeled_tenant_namespace.name}")
                return


@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-evalhub-mt-switch-to-st"})],
    indirect=True,
)
@pytest.mark.tier2
@pytest.mark.ai_safety
class TestEvalHubModeSwitchMultiToSingle:
    """EvalHub starts in multi mode; patching spec.tenancy to single creates Roles
    and removes cross-namespace RoleBindings from the tenant namespace.
    """

    def test_tenant_rbac_exists_in_multi_mode(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        labeled_tenant_namespace: Namespace,
        evalhub_mt_for_switch: EvalHub,
        evalhub_mt_switch_deployment: Deployment,
    ) -> None:
        """Given: a multi-tenancy EvalHub with a labeled tenant namespace.

        When: tenant RBAC is inspected before switching to single mode.

        Then: a jobs-writer RoleBinding exists in the tenant namespace.
        """
        cr_name = evalhub_mt_for_switch.name

        def _rbac_ready() -> bool:
            rbs = list(RoleBinding.get(client=admin_client, namespace=labeled_tenant_namespace.name))
            return any(
                rb.instance.roleRef.name == EVALHUB_JOBS_WRITER_CLUSTERROLE and rb.name.startswith(cr_name)
                for rb in rbs
            )

        for ready in TimeoutSampler(wait_timeout=120, sleep=5, func=_rbac_ready):
            if ready:
                return

    def test_switch_to_single_creates_roles(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_mt_for_switch: EvalHub,
    ) -> None:
        """Given: a multi-tenancy EvalHub with cross-namespace RBAC in a tenant namespace.

        When: spec.tenancy is patched to single.

        Then: evalhub-tenant-admin, evalhub-user, and evalhub-tenant-admin-binding
        are created in the instance namespace.
        """
        ResourceEditor(patches={evalhub_mt_for_switch: {"spec": {"tenancy": EVALHUB_TENANCY_SINGLE}}}).update()
        LOGGER.info(f"Patched '{evalhub_mt_for_switch.name}' spec.tenancy → single")

        def _roles_exist() -> bool:
            admin_role = Role(client=admin_client, name=EVALHUB_TENANT_ADMIN_ROLE_NAME, namespace=model_namespace.name)
            user_role = Role(client=admin_client, name=EVALHUB_USER_ROLE_NAME, namespace=model_namespace.name)
            binding = RoleBinding(
                client=admin_client, name=EVALHUB_TENANT_ADMIN_BINDING_NAME, namespace=model_namespace.name
            )
            return admin_role.exists and user_role.exists and binding.exists

        for exists in TimeoutSampler(wait_timeout=120, sleep=5, func=_roles_exist):
            if exists:
                LOGGER.info("Convenience Roles created after switch to single mode")
                return

    def test_switch_to_single_removes_tenant_rbac(
        self,
        admin_client: DynamicClient,
        labeled_tenant_namespace: Namespace,
        evalhub_mt_for_switch: EvalHub,
    ) -> None:
        """Given: a multi-tenancy EvalHub switched to single mode.

        When: the operator cleans up cross-namespace RoleBindings.

        Then: jobs-writer and job-config RoleBindings are removed from the tenant namespace.
        """
        cr_name = evalhub_mt_for_switch.name

        def _tenant_rbac_gone() -> bool:
            rbs = list(RoleBinding.get(client=admin_client, namespace=labeled_tenant_namespace.name))
            has_writer = any(
                rb.instance.roleRef.name == EVALHUB_JOBS_WRITER_CLUSTERROLE and rb.name.startswith(cr_name)
                for rb in rbs
            )
            has_config = any(
                rb.instance.roleRef.name == EVALHUB_JOB_CONFIG_CLUSTERROLE and rb.name.startswith(cr_name) for rb in rbs
            )
            return not has_writer and not has_config

        for gone in TimeoutSampler(wait_timeout=120, sleep=5, func=_tenant_rbac_gone):
            if gone:
                LOGGER.info(f"Cross-namespace RoleBindings cleaned from {labeled_tenant_namespace.name}")
                return


@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-evalhub-st-cm-hotmount"})],
    indirect=True,
)
@pytest.mark.tier2
@pytest.mark.ai_safety
class TestEvalHubTenantConfigMapHotMount:
    """A ConfigMap labeled evalhub-provider-type=tenant triggers a Deployment rollout.

    The operator watches for ConfigMaps with the provider-type=tenant label in the
    instance namespace. When one is created (or updated), the operator reconciles and
    adds the ConfigMap to the Deployment's projected volumes, causing a new rollout.
    """

    def test_tenant_labeled_cm_triggers_deployment_update(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_st_cr: SingleTenantEvalHub,
        evalhub_st_deployment: Deployment,
    ) -> None:
        """Given: a single-tenancy EvalHub with a ready Deployment.

        When: a ConfigMap labelled evalhub-provider-type=tenant is created in the instance namespace.

        Then: the Deployment metadata.generation increments (rollout triggered).
        """
        initial_generation = evalhub_st_deployment.instance.metadata.generation or 0
        cm_name = "evalhub-st-test-tenant-provider"

        with ConfigMap(
            client=admin_client,
            name=cm_name,
            namespace=model_namespace.name,
            data={"provider.yaml": "id: test-tenant-provider\ntype: lm_evaluation_harness\n"},
            label={EVALHUB_TENANT_PROVIDER_LABEL_KEY: EVALHUB_TENANT_PROVIDER_LABEL_VAL},
            wait_for_resource=True,
        ):
            LOGGER.info(f"Created tenant-labeled ConfigMap '{cm_name}' in {model_namespace.name}")

            def _deployment_updated() -> bool:
                current_gen = evalhub_st_deployment.instance.metadata.generation or 0
                return current_gen > initial_generation

            for updated in TimeoutSampler(wait_timeout=120, sleep=5, func=_deployment_updated):
                if updated:
                    LOGGER.info(
                        f"Deployment generation incremented: "
                        f"{initial_generation} → "
                        f"{evalhub_st_deployment.instance.metadata.generation}"
                    )
                    return


@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-evalhub-st-deletion"})],
    indirect=True,
)
@pytest.mark.tier2
@pytest.mark.ai_safety
class TestEvalHubSingleTenancyDeletion:
    """Deleting a single-tenant EvalHub CR causes the owner-ref'd RBAC objects to be GC'd.

    The operator sets controller owner references on evalhub-tenant-admin,
    evalhub-user, and evalhub-tenant-admin-binding. When the CR is deleted
    the Kubernetes GC removes these objects automatically — no finalizer needed.
    """

    def test_roles_gc_after_cr_deletion(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        evalhub_st_cr: SingleTenantEvalHub,
        evalhub_st_deployment: Deployment,
    ) -> None:
        """Given: a single-tenancy EvalHub with owner-referenced convenience Roles and RoleBinding.

        When: the EvalHub CR is deleted.

        Then: evalhub-tenant-admin, evalhub-user, and evalhub-tenant-admin-binding are garbage-collected.
        """
        # Verify Roles exist before deletion (pre-condition)
        admin_role = Role(
            client=admin_client,
            name=EVALHUB_TENANT_ADMIN_ROLE_NAME,
            namespace=model_namespace.name,
        )
        assert admin_role.exists, f"Pre-condition: Role '{EVALHUB_TENANT_ADMIN_ROLE_NAME}' not found before deletion"

        # Delete the EvalHub CR directly (context manager will also delete on exit,
        # but we need to trigger deletion inside the test body to then poll for GC)
        evalhub_st_cr.delete(wait=False)
        LOGGER.info(f"Deleted EvalHub CR '{evalhub_st_cr.name}' — waiting for RBAC GC")

        def _rbac_gc_complete() -> bool:
            a = Role(client=admin_client, name=EVALHUB_TENANT_ADMIN_ROLE_NAME, namespace=model_namespace.name)
            u = Role(client=admin_client, name=EVALHUB_USER_ROLE_NAME, namespace=model_namespace.name)
            b = RoleBinding(client=admin_client, name=EVALHUB_TENANT_ADMIN_BINDING_NAME, namespace=model_namespace.name)
            return not a.exists and not u.exists and not b.exists

        for done in TimeoutSampler(wait_timeout=120, sleep=5, func=_rbac_gc_complete):
            if done:
                LOGGER.info("Owner-ref'd RBAC objects garbage-collected after CR deletion")
                return
