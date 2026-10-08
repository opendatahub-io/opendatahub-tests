"""Tenant namespace discovery reconciliation tests (RHOAIENG-98878)."""

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.maas_model_ref import MaaSModelRef

from tests.ai_gateway.models_as_a_service.maas_subscription.utils import create_maas_subscription
from tests.ai_gateway.models_as_a_service.multitenancy.aitenant.utils import (
    FINALIZER_AUTH_POLICY,
    FINALIZER_SUBSCRIPTION,
    SUBSCRIPTION_RECONCILED_PHASES,
    AITenantTestContext,
    assert_maas_resource_stays_unreconciled,
    prepare_discovered_tenant_namespace,
    read_maas_resource_finalizers,
    remove_discovery_namespace_labels,
    synthetic_discovery_tenant_namespace,
    verify_tenant_namespace_discovery_labels_present,
    wait_for_maas_auth_policy_active,
    wait_for_maas_model_ref_discovered,
    wait_for_maas_resource_finalizer,
    wait_for_maas_resource_phase,
    wait_until_maas_controller_stops_reconciling_discovery_namespace,
)
from utilities.general import generate_random_name
from utilities.resources.maa_s_auth_policy import MaaSAuthPolicy

LOGGER = structlog.get_logger(name=__name__)

AUTHENTICATED_GROUP = "system:authenticated"


def _tinyllama_model_ref(maas_model_tinyllama_free: MaaSModelRef) -> tuple[str, str]:
    return maas_model_tinyllama_free.name, maas_model_tinyllama_free.namespace


@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "aitenant_infra_namespace",
    "tenant_namespace_discovery_prerequisites",
    "maas_model_tinyllama_free",
)
class TestTenantNamespaceDiscovery:
    @pytest.mark.tier1
    def test_labeled_namespace_reconciles_maas_auth_policy(
        self,
        admin_client: DynamicClient,
        teardown_resources: bool,
        maas_model_tinyllama_free: MaaSModelRef,
    ) -> None:
        """Given a discovery-labeled namespace with MaasTenantConfig,
        when a MaaSAuthPolicy is created,
        then the controller adds its finalizer and sets status.phase Active.
        """
        model_name, model_namespace = _tinyllama_model_ref(maas_model_tinyllama_free=maas_model_tinyllama_free)
        with (
            synthetic_discovery_tenant_namespace(
                admin_client=admin_client,
                teardown=teardown_resources,
            ) as case,
            MaaSAuthPolicy(
                client=admin_client,
                name=case["policy_name"],
                namespace=case["tenant_namespace_name"],
                model_refs=[{"name": model_name, "namespace": model_namespace}],
                subjects={"groups": [{"name": AUTHENTICATED_GROUP}]},
                teardown=teardown_resources,
                wait_for_resource=True,
            ) as auth_policy,
        ):
            wait_for_maas_auth_policy_active(auth_policy=auth_policy)

    @pytest.mark.tier1
    def test_labeled_namespace_reconciles_maas_subscription(
        self,
        admin_client: DynamicClient,
        teardown_resources: bool,
        maas_model_tinyllama_free: MaaSModelRef,
    ) -> None:
        """Given a discovery-labeled namespace with MaasTenantConfig,
        when a MaaSSubscription is created,
        then the controller reconciles it to an Active or Degraded phase.
        """
        model_name, model_namespace = _tinyllama_model_ref(maas_model_tinyllama_free=maas_model_tinyllama_free)
        with (
            synthetic_discovery_tenant_namespace(
                admin_client=admin_client,
                teardown=teardown_resources,
            ) as case,
            create_maas_subscription(
                admin_client=admin_client,
                subscription_namespace=case["tenant_namespace_name"],
                subscription_name=case["subscription_name"],
                owner_group_name=AUTHENTICATED_GROUP,
                model_name=model_name,
                model_namespace=model_namespace,
                tokens_per_minute=100,
                teardown=teardown_resources,
                wait_for_resource=True,
            ) as subscription,
        ):
            wait_for_maas_resource_finalizer(
                resource=subscription,
                expected_finalizer=FINALIZER_SUBSCRIPTION,
            )
            phase = wait_for_maas_resource_phase(
                resource=subscription,
                expected_phases=SUBSCRIPTION_RECONCILED_PHASES,
            )
            assert phase in SUBSCRIPTION_RECONCILED_PHASES

    @pytest.mark.tier1
    def test_labeled_namespace_reconciles_maas_model_ref(
        self,
        admin_client: DynamicClient,
        teardown_resources: bool,
        maas_model_tinyllama_free: MaaSModelRef,
    ) -> None:
        """Given a discovery-labeled namespace with MaasTenantConfig,
        when a tenant-local MaaSModelRef references a shared model,
        then maas-controller adds its finalizer and sets status.phase.
        """
        model_name, model_namespace = _tinyllama_model_ref(maas_model_tinyllama_free=maas_model_tinyllama_free)
        with (
            synthetic_discovery_tenant_namespace(
                admin_client=admin_client,
                teardown=teardown_resources,
            ) as case,
            MaaSModelRef(
                client=admin_client,
                name=case["model_ref_name"],
                namespace=case["tenant_namespace_name"],
                model_ref={
                    "name": model_name,
                    "namespace": model_namespace,
                    "kind": "LLMInferenceService",
                },
                teardown=teardown_resources,
                wait_for_resource=True,
            ) as tenant_model_ref,
        ):
            wait_for_maas_model_ref_discovered(model_ref=tenant_model_ref)
            LOGGER.info(
                f"MaaSModelRef '{case['tenant_namespace_name']}/{case['model_ref_name']}' "
                f"reconciled in discovery namespace"
            )

    @pytest.mark.tier1
    def test_unlabeled_namespace_maas_crs_ignored(
        self,
        admin_client: DynamicClient,
        teardown_resources: bool,
        maas_model_tinyllama_free: MaaSModelRef,
    ) -> None:
        """Given a namespace with MaasTenantConfig but no discovery labels,
        when MaaSAuthPolicy and MaaSSubscription are created,
        then maas-controller does not reconcile them.
        """
        model_name, model_namespace = _tinyllama_model_ref(maas_model_tinyllama_free=maas_model_tinyllama_free)
        with (
            synthetic_discovery_tenant_namespace(
                admin_client=admin_client,
                teardown=teardown_resources,
                discovery_labels_applied=False,
            ) as case,
            MaaSAuthPolicy(
                client=admin_client,
                name=case["policy_name"],
                namespace=case["tenant_namespace_name"],
                model_refs=[{"name": model_name, "namespace": model_namespace}],
                subjects={"groups": [{"name": AUTHENTICATED_GROUP}]},
                teardown=teardown_resources,
                wait_for_resource=True,
            ) as auth_policy,
            create_maas_subscription(
                admin_client=admin_client,
                subscription_namespace=case["tenant_namespace_name"],
                subscription_name=case["subscription_name"],
                owner_group_name=AUTHENTICATED_GROUP,
                model_name=model_name,
                model_namespace=model_namespace,
                tokens_per_minute=100,
                teardown=teardown_resources,
                wait_for_resource=True,
            ) as subscription,
        ):
            unreconciled_timeout = 60
            assert_maas_resource_stays_unreconciled(
                resource=auth_policy,
                forbidden_finalizer=FINALIZER_AUTH_POLICY,
                timeout=unreconciled_timeout,
            )
            assert_maas_resource_stays_unreconciled(
                resource=subscription,
                forbidden_finalizer=FINALIZER_SUBSCRIPTION,
                timeout=unreconciled_timeout,
            )

    @pytest.mark.tier2
    def test_aitenant_bootstrap_tenant_namespace_reconciles_maas_auth_policy(
        self,
        admin_client: DynamicClient,
        teardown_resources: bool,
        aitenant_for_test: AITenantTestContext,
        maas_model_tinyllama_free: MaaSModelRef,
    ) -> None:
        """Given a Ready AITenant bootstrap (discovery labels + MaasTenantConfig on tenant NS),
        when a MaaSAuthPolicy is created in that namespace,
        then maas-controller reconciles it.
        """
        tenant_namespace_name = aitenant_for_test["tenant_namespace_name"]
        policy_name = f"e2e-aitenant-discovery-{generate_random_name()[:8]}"
        model_name, model_namespace = _tinyllama_model_ref(maas_model_tinyllama_free=maas_model_tinyllama_free)
        verify_tenant_namespace_discovery_labels_present(
            admin_client=admin_client,
            tenant_namespace_name=tenant_namespace_name,
        )
        with MaaSAuthPolicy(
            client=admin_client,
            name=policy_name,
            namespace=tenant_namespace_name,
            model_refs=[{"name": model_name, "namespace": model_namespace}],
            subjects={"groups": [{"name": AUTHENTICATED_GROUP}]},
            teardown=teardown_resources,
            wait_for_resource=True,
        ) as auth_policy:
            wait_for_maas_auth_policy_active(auth_policy=auth_policy)

    @pytest.mark.tier2
    def test_maas_crs_reconcile_after_discovery_labels_applied(
        self,
        admin_client: DynamicClient,
        teardown_resources: bool,
        maas_model_tinyllama_free: MaaSModelRef,
    ) -> None:
        """Given MaaSAuthPolicy exists before discovery labels,
        when discovery labels are applied to the namespace,
        then the controller starts reconciliation.
        """
        model_name, model_namespace = _tinyllama_model_ref(maas_model_tinyllama_free=maas_model_tinyllama_free)
        with (
            synthetic_discovery_tenant_namespace(
                admin_client=admin_client,
                teardown=teardown_resources,
                discovery_labels_applied=False,
            ) as case,
            MaaSAuthPolicy(
                client=admin_client,
                name=case["policy_name"],
                namespace=case["tenant_namespace_name"],
                model_refs=[{"name": model_name, "namespace": model_namespace}],
                subjects={"groups": [{"name": AUTHENTICATED_GROUP}]},
                teardown=teardown_resources,
                wait_for_resource=True,
            ) as auth_policy,
        ):
            assert_maas_resource_stays_unreconciled(
                resource=auth_policy,
                forbidden_finalizer=FINALIZER_AUTH_POLICY,
                timeout=30,
            )
            prepare_discovered_tenant_namespace(
                admin_client=admin_client,
                tenant_namespace_name=case["tenant_namespace_name"],
                tenant_label_name=case["tenant_label_name"],
            )
            wait_for_maas_auth_policy_active(auth_policy=auth_policy)

    @pytest.mark.tier2
    def test_label_removal_stops_new_reconciliation(
        self,
        admin_client: DynamicClient,
        teardown_resources: bool,
        maas_model_tinyllama_free: MaaSModelRef,
    ) -> None:
        """Given discovery labels are removed from a tenant namespace,
        when a new MaaSAuthPolicy is created,
        then maas-controller does not reconcile it.
        """
        model_name, model_namespace = _tinyllama_model_ref(maas_model_tinyllama_free=maas_model_tinyllama_free)
        with synthetic_discovery_tenant_namespace(
            admin_client=admin_client,
            teardown=teardown_resources,
        ) as case:
            post_label_policy_name = f"e2e-post-label-{case['suffix']}"
            with MaaSAuthPolicy(
                client=admin_client,
                name=case["policy_name"],
                namespace=case["tenant_namespace_name"],
                model_refs=[{"name": model_name, "namespace": model_namespace}],
                subjects={"groups": [{"name": AUTHENTICATED_GROUP}]},
                teardown=teardown_resources,
                wait_for_resource=True,
            ) as seed_policy:
                wait_for_maas_resource_finalizer(
                    resource=seed_policy,
                    expected_finalizer=FINALIZER_AUTH_POLICY,
                )
                remove_discovery_namespace_labels(
                    admin_client=admin_client,
                    tenant_namespace_name=case["tenant_namespace_name"],
                )
                wait_until_maas_controller_stops_reconciling_discovery_namespace(
                    admin_client=admin_client,
                    tenant_namespace_name=case["tenant_namespace_name"],
                    model_name=model_name,
                    model_namespace=model_namespace,
                    forbidden_finalizer=FINALIZER_AUTH_POLICY,
                    teardown=teardown_resources,
                )
                with MaaSAuthPolicy(
                    client=admin_client,
                    name=post_label_policy_name,
                    namespace=case["tenant_namespace_name"],
                    model_refs=[{"name": model_name, "namespace": model_namespace}],
                    subjects={"groups": [{"name": AUTHENTICATED_GROUP}]},
                    teardown=teardown_resources,
                    wait_for_resource=True,
                ) as new_policy:
                    assert_maas_resource_stays_unreconciled(
                        resource=new_policy,
                        forbidden_finalizer=FINALIZER_AUTH_POLICY,
                        timeout=60,
                    )
                    finalizers = read_maas_resource_finalizers(resource=new_policy)
                    assert FINALIZER_AUTH_POLICY not in finalizers
