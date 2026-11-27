"""Legacy Tenant/default-tenant cleanup after MaasTenantConfig migration (RHOAIENG-98880)."""

import pytest
import requests
from kubernetes.dynamic import DynamicClient

from tests.ai_gateway.models_as_a_service.multitenancy.aitenant.utils import (
    LEGACY_TENANT_MIGRATION_API_KEYS_DAYS_LEGACY,
    LEGACY_TENANT_MIGRATION_API_KEYS_DAYS_MISMATCH,
    build_legacy_tenant_migration_telemetry,
    build_legacy_tenant_migration_telemetry_disabled,
    legacy_tenant_migration_ready_aitenant,
    prepare_tenant_maas_api_gateway_access,
    run_legacy_tenant_migration_e2e,
    tenant_maas_api_base_url_for_gateway_name,
    verify_legacy_tenant_cleanup_migration_outcome,
    verify_maas_api_key_and_models_after_legacy_migration,
)
from tests.ai_gateway.models_as_a_service.multitenancy.utils import tenant_gateway_external_route


@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "aitenant_infra_namespace",
    "legacy_tenant_migration_prerequisites",
)
class TestLegacyTenantCleanupAfterMigration:
    """Verify AITenant reconciler legacy Tenant cleanup after MaasTenantConfig migration."""

    @pytest.mark.tier2
    def test_legacy_tenant_removed_when_maas_tenant_config_matches_after_aitenant_migration(
        self,
        admin_client: DynamicClient,
        aitenant_infra_namespace: str,
        teardown_resources: bool,
    ) -> None:
        """Given legacy Tenant with gatewayRef, externalOIDC, apiKeys, and telemetry in ai-tenant-* namespace,
        when AITenant reconciles and controller creates aligned MaasTenantConfig from legacy,
        then legacy Tenant is deleted, MTC copies apiKeys/telemetry, gateway/OIDC move to AITenant.spec,
        and models-as-a-service default-tenant MaasTenantConfig stays Ready.
        """
        run_legacy_tenant_migration_e2e(
            admin_client=admin_client,
            aitenant_infra_namespace=aitenant_infra_namespace,
            teardown=teardown_resources,
            maas_tenant_config_api_keys_max_expiration_days=None,
            expect_legacy_deleted=True,
            outcome_log_message="Legacy Tenant cleanup succeeded",
            legacy_gateway_and_external_oidc=True,
            expected_mtc_telemetry=build_legacy_tenant_migration_telemetry(),
            verify_aitenant_gateway_and_oidc=True,
            verify_cluster_default_tenant=True,
        )

    @pytest.mark.tier2
    def test_legacy_tenant_retained_when_maas_tenant_config_api_keys_differ(
        self,
        admin_client: DynamicClient,
        aitenant_infra_namespace: str,
        teardown_resources: bool,
    ) -> None:
        """Given legacy Tenant apiKeys and MaasTenantConfig apiKeys disagree before reconcile,
        when AITenant reconciles the namespace,
        then legacy Tenant/default-tenant is retained and MaasTenantConfig stays Ready.
        """
        run_legacy_tenant_migration_e2e(
            admin_client=admin_client,
            aitenant_infra_namespace=aitenant_infra_namespace,
            teardown=teardown_resources,
            maas_tenant_config_api_keys_max_expiration_days=LEGACY_TENANT_MIGRATION_API_KEYS_DAYS_MISMATCH,
            expect_legacy_deleted=False,
            outcome_log_message="Legacy Tenant retained on apiKeys mismatch",
        )

    @pytest.mark.tier2
    def test_legacy_tenant_retained_when_maas_tenant_config_telemetry_differ(
        self,
        admin_client: DynamicClient,
        aitenant_infra_namespace: str,
        teardown_resources: bool,
    ) -> None:
        """Given legacy Tenant and MaasTenantConfig share apiKeys but telemetry differs before reconcile,
        when AITenant reconciles the namespace,
        then legacy Tenant/default-tenant is retained and MaasTenantConfig stays Ready.
        """
        run_legacy_tenant_migration_e2e(
            admin_client=admin_client,
            aitenant_infra_namespace=aitenant_infra_namespace,
            teardown=teardown_resources,
            maas_tenant_config_api_keys_max_expiration_days=LEGACY_TENANT_MIGRATION_API_KEYS_DAYS_LEGACY,
            maas_tenant_config_telemetry=build_legacy_tenant_migration_telemetry_disabled(),
            expect_legacy_deleted=False,
            outcome_log_message="Legacy Tenant retained on telemetry mismatch",
        )

    @pytest.mark.tier2
    @pytest.mark.usefixtures("authorino_tls_configured")
    def test_maas_api_key_and_models_after_legacy_tenant_removed(
        self,
        admin_client: DynamicClient,
        aitenant_infra_namespace: str,
        maas_api_infra_namespace: str,
        maas_host: str,
        request_session_http: requests.Session,
        ocp_token_for_actor: str,
        legacy_tenant_migration_api_key_governance,
        teardown_resources: bool,
    ) -> None:
        """Given legacy Tenant was removed after aligned MaasTenantConfig migration,
        when a new API key is minted on the per-tenant maas-api URL (tenant Gateway, not models-as-a-service),
        then GET /v1/models with that key returns 200.

        Does not prove a key created before migration still works. Requires tenant MaaSSubscription for mint.
        """
        with legacy_tenant_migration_ready_aitenant(
            admin_client=admin_client,
            aitenant_infra_namespace=aitenant_infra_namespace,
            maas_api_infra_namespace=maas_api_infra_namespace,
            teardown=teardown_resources,
            legacy_api_keys_max_expiration_days=LEGACY_TENANT_MIGRATION_API_KEYS_DAYS_LEGACY,
            maas_tenant_config_api_keys_max_expiration_days=None,
        ) as test_context:
            verify_legacy_tenant_cleanup_migration_outcome(
                admin_client=admin_client,
                test_context=test_context,
                expect_legacy_deleted=True,
                expected_mtc_telemetry=build_legacy_tenant_migration_telemetry(),
            )
            gateway_name, gateway_namespace = prepare_tenant_maas_api_gateway_access(
                admin_client=admin_client,
                test_context=test_context,
                maas_api_infra_namespace=maas_api_infra_namespace,
            )
            tenant_maas_api_base_url = tenant_maas_api_base_url_for_gateway_name(
                gateway_name=gateway_name,
                maas_host=maas_host,
            )
            with (
                tenant_gateway_external_route(
                    admin_client=admin_client,
                    gateway_name=gateway_name,
                    maas_host=maas_host,
                    teardown=teardown_resources,
                ),
                legacy_tenant_migration_api_key_governance(
                    test_context,
                    gateway_name,
                    gateway_namespace,
                ) as subscription_name,
            ):
                verify_maas_api_key_and_models_after_legacy_migration(
                    test_context=test_context,
                    tenant_maas_api_base_url=tenant_maas_api_base_url,
                    request_session_http=request_session_http,
                    ocp_user_token=ocp_token_for_actor,
                    subscription_name=subscription_name,
                )
