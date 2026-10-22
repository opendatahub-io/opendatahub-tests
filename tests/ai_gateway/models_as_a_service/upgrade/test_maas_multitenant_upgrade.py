"""MaaS multitenant upgrade baseline tests (RHOAIENG-100389)."""

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace

from tests.ai_gateway.models_as_a_service.multitenancy.utils import (
    verify_maas_api_deployment_for_aitenant,
)
from tests.ai_gateway.models_as_a_service.praxis.utils import verify_praxis_payload_processing_active_for_aitenant
from tests.ai_gateway.models_as_a_service.upgrade.utils import (
    MultitenantUpgradeTenantRecord,
    aitenant_for_multitenant_upgrade_record,
    assert_multitenant_upgrade_cross_tenant_inference_rejected,
    assert_multitenant_upgrade_inference_isolation_for_pair,
    assert_multitenant_upgrade_own_inference_succeeds,
    assert_multitenant_upgrade_rate_limit_isolation,
    multitenant_upgrade_inference_chat_payload,
    verify_multitenant_upgrade_cross_tenant_api_key_get_isolation,
    verify_multitenant_upgrade_maas_api_db_migration_logs_clean,
    verify_multitenant_upgrade_maastenantconfig_active,
    verify_multitenant_upgrade_models_list_probe,
    verify_multitenant_upgrade_tenant_governance_survived,
)
from tests.ai_gateway.models_as_a_service.utils import verify_aitenant_ready

LOGGER = structlog.get_logger(name=__name__)

MULTITENANT_UPGRADE_TENANT_OWNER_INDEX = [
    pytest.param(0, id="test_tenant_a"),
    pytest.param(1, id="test_tenant_b"),
]


def _tenant_pair(
    tenant_records: list[MultitenantUpgradeTenantRecord],
) -> tuple[MultitenantUpgradeTenantRecord, MultitenantUpgradeTenantRecord]:
    assert len(tenant_records) == 2, f"Expected 2 multitenant upgrade tenants, got {len(tenant_records)}"
    return tenant_records[0], tenant_records[1]


@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "authorino_tls_configured",
    "maas_gateway_api",
    "aitenant_infra_namespace",
    "maas_multitenant_upgrade_namespace",
    "maas_multitenant_upgrade_tenant_records",
    "capture_maas_multitenant_upgrade_baseline",
)
class TestPreUpgradeMultitenantMaaS:
    """Provision two Ready AITenants with models, subscriptions, and API keys before upgrade."""

    @pytest.mark.pre_upgrade
    def test_multitenant_aitenants_ready_pre_upgrade(
        self,
        admin_client: DynamicClient,
        aitenant_infra_namespace: str,
        maas_multitenant_upgrade_tenant_records: list[MultitenantUpgradeTenantRecord],
    ) -> None:
        """Given two provisioned AITenants in the infra namespace, when checking status pre-upgrade,
        then both report Ready with phase Active.
        """
        for tenant_record in maas_multitenant_upgrade_tenant_records:
            aitenant = aitenant_for_multitenant_upgrade_record(
                admin_client=admin_client,
                aitenant_infra_namespace=aitenant_infra_namespace,
                tenant_record=tenant_record,
            )
            verify_aitenant_ready(aitenant=aitenant)
            verify_multitenant_upgrade_maastenantconfig_active(
                admin_client=admin_client,
                tenant_record=tenant_record,
            )

    @pytest.mark.pre_upgrade
    def test_multitenant_api_key_cross_tenant_rejected_pre_upgrade(
        self,
        request_session_http: requests.Session,
        maas_host: str,
        current_client_token: str,
        maas_multitenant_upgrade_tenant_records: list[MultitenantUpgradeTenantRecord],
    ) -> None:
        """Given API keys for both tenants, when fetching each key via the other tenant maas-api,
        then both foreign gateways return 404.
        """
        verify_multitenant_upgrade_cross_tenant_api_key_get_isolation(
            request_session_http=request_session_http,
            maas_host=maas_host,
            ocp_user_token=current_client_token,
            tenant_records=maas_multitenant_upgrade_tenant_records,
        )

    @pytest.mark.pre_upgrade
    def test_multitenant_inference_pre_upgrade(
        self,
        admin_client: DynamicClient,
        maas_multitenant_upgrade_namespace: Namespace,
        maas_host: str,
        request_session_http: requests.Session,
        maas_multitenant_upgrade_tenant_records: list[MultitenantUpgradeTenantRecord],
    ) -> None:
        """Given persisted API keys, when posting inference on each tenant gateway pre-upgrade,
        then every tenant returns 200 on its own gateway.
        """
        persist_namespace = maas_multitenant_upgrade_namespace.name
        chat_payload = multitenant_upgrade_inference_chat_payload()
        for tenant_record in maas_multitenant_upgrade_tenant_records:
            assert_multitenant_upgrade_own_inference_succeeds(
                admin_client=admin_client,
                persist_namespace=persist_namespace,
                maas_host=maas_host,
                request_session_http=request_session_http,
                tenant_record=tenant_record,
                chat_payload=chat_payload,
            )

    @pytest.mark.parametrize("owner_index", MULTITENANT_UPGRADE_TENANT_OWNER_INDEX)
    @pytest.mark.pre_upgrade
    def test_multitenant_cross_tenant_inference_rejected_pre_upgrade(
        self,
        owner_index: int,
        admin_client: DynamicClient,
        maas_multitenant_upgrade_namespace: Namespace,
        maas_host: str,
        request_session_http: requests.Session,
        maas_multitenant_upgrade_tenant_records: list[MultitenantUpgradeTenantRecord],
    ) -> None:
        """Given persisted API keys pre-upgrade, when posting inference with tenant A's key on tenant B gateway,
        then the foreign gateway returns 401 or 403 (symmetric for tenant B key on tenant A gateway).
        """
        tenant_a_record, tenant_b_record = _tenant_pair(
            tenant_records=maas_multitenant_upgrade_tenant_records,
        )
        owner_record = tenant_a_record if owner_index == 0 else tenant_b_record
        peer_record = tenant_b_record if owner_index == 0 else tenant_a_record
        persist_namespace = maas_multitenant_upgrade_namespace.name
        chat_payload = multitenant_upgrade_inference_chat_payload()
        assert_multitenant_upgrade_cross_tenant_inference_rejected(
            admin_client=admin_client,
            persist_namespace=persist_namespace,
            maas_host=maas_host,
            request_session_http=request_session_http,
            owner_tenant_record=owner_record,
            foreign_tenant_record=peer_record,
            chat_payload=chat_payload,
        )

    @pytest.mark.pre_upgrade
    def test_multitenant_models_probe_pre_upgrade(
        self,
        admin_client: DynamicClient,
        maas_multitenant_upgrade_namespace: Namespace,
        maas_host: str,
        request_session_http: requests.Session,
        maas_multitenant_upgrade_tenant_records: list[MultitenantUpgradeTenantRecord],
    ) -> None:
        """Given persisted API keys, when calling GET /v1/models on each tenant maas-api pre-upgrade,
        then each tenant lists models successfully.
        """
        persist_namespace = maas_multitenant_upgrade_namespace.name
        for tenant_record in maas_multitenant_upgrade_tenant_records:
            verify_multitenant_upgrade_models_list_probe(
                admin_client=admin_client,
                persist_namespace=persist_namespace,
                maas_host=maas_host,
                request_session_http=request_session_http,
                tenant_record=tenant_record,
            )


@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "authorino_tls_configured",
    "maas_gateway_api",
    "aitenant_infra_namespace",
    "maas_multitenant_upgrade_namespace",
    "maas_multitenant_upgrade_tenant_records",
)
class TestPostUpgradeMultitenantMaaS:
    """Validate multitenant MaaS state after operator upgrade."""

    @pytest.mark.post_upgrade
    def test_multitenant_aitenants_ready_post_upgrade(
        self,
        admin_client: DynamicClient,
        aitenant_infra_namespace: str,
        maas_multitenant_upgrade_tenant_records: list[MultitenantUpgradeTenantRecord],
    ) -> None:
        """Given pre-upgrade multitenant baseline, when upgrade completes,
        then every AITenant in the infra namespace remains Ready.
        """
        for tenant_record in maas_multitenant_upgrade_tenant_records:
            aitenant = aitenant_for_multitenant_upgrade_record(
                admin_client=admin_client,
                aitenant_infra_namespace=aitenant_infra_namespace,
                tenant_record=tenant_record,
            )
            verify_aitenant_ready(aitenant=aitenant)
            verify_multitenant_upgrade_maastenantconfig_active(
                admin_client=admin_client,
                tenant_record=tenant_record,
            )

    @pytest.mark.post_upgrade
    def test_multitenant_maas_api_and_payload_processing_ready_post_upgrade(
        self,
        admin_client: DynamicClient,
        aitenant_infra_namespace: str,
        maas_api_infra_namespace: str,
        maas_multitenant_upgrade_tenant_records: list[MultitenantUpgradeTenantRecord],
    ) -> None:
        """Given multitenant baseline, when upgrade completes,
        then each maas-api-{tenant} Deployment is Available and Praxis payload processing is active.
        """
        for tenant_record in maas_multitenant_upgrade_tenant_records:
            verify_maas_api_deployment_for_aitenant(
                admin_client=admin_client,
                api_namespace=maas_api_infra_namespace,
                aitenant_name=tenant_record["aitenant_name"],
                tenant_namespace_name=tenant_record["tenant_namespace_name"],
            )
            aitenant = aitenant_for_multitenant_upgrade_record(
                admin_client=admin_client,
                aitenant_infra_namespace=aitenant_infra_namespace,
                tenant_record=tenant_record,
            )
            verify_praxis_payload_processing_active_for_aitenant(
                admin_client=admin_client,
                aitenant=aitenant,
            )

    @pytest.mark.post_upgrade
    def test_multitenant_maas_api_db_migration_logs_clean_post_upgrade(
        self,
        admin_client: DynamicClient,
        maas_api_infra_namespace: str,
        maas_multitenant_upgrade_tenant_records: list[MultitenantUpgradeTenantRecord],
    ) -> None:
        """Given multitenant maas-api Deployments are Ready post-upgrade,
        then each per-tenant maas-api Pod log shows no dirty or failed DB migration markers.
        """
        verify_multitenant_upgrade_maas_api_db_migration_logs_clean(
            admin_client=admin_client,
            maas_api_infra_namespace=maas_api_infra_namespace,
            tenant_records=maas_multitenant_upgrade_tenant_records,
        )

    @pytest.mark.post_upgrade
    def test_multitenant_tenant_governance_survived_post_upgrade(
        self,
        admin_client: DynamicClient,
        maas_multitenant_upgrade_tenant_records: list[MultitenantUpgradeTenantRecord],
    ) -> None:
        """Given pre-upgrade tenant governance baseline, when upgrade completes,
        then each tenant MaaSModelRef, MaaSAuthPolicy, and MaaSSubscription generation is unchanged.
        """
        for tenant_record in maas_multitenant_upgrade_tenant_records:
            verify_multitenant_upgrade_tenant_governance_survived(
                admin_client=admin_client,
                tenant_record=tenant_record,
            )

    @pytest.mark.post_upgrade
    def test_multitenant_api_key_cross_tenant_rejected_post_upgrade(
        self,
        request_session_http: requests.Session,
        maas_host: str,
        current_client_token: str,
        maas_multitenant_upgrade_tenant_records: list[MultitenantUpgradeTenantRecord],
    ) -> None:
        """Given persisted API keys after upgrade, when fetching each key via the other tenant maas-api,
        then both foreign gateways still return 404.
        """
        verify_multitenant_upgrade_cross_tenant_api_key_get_isolation(
            request_session_http=request_session_http,
            maas_host=maas_host,
            ocp_user_token=current_client_token,
            tenant_records=maas_multitenant_upgrade_tenant_records,
        )

    @pytest.mark.post_upgrade
    def test_multitenant_models_probe_post_upgrade(
        self,
        admin_client: DynamicClient,
        maas_multitenant_upgrade_namespace: Namespace,
        maas_host: str,
        request_session_http: requests.Session,
        maas_multitenant_upgrade_tenant_records: list[MultitenantUpgradeTenantRecord],
    ) -> None:
        """Given persisted API keys after upgrade, when calling GET /v1/models on each tenant maas-api,
        then each tenant still lists models successfully.
        """
        persist_namespace = maas_multitenant_upgrade_namespace.name
        for tenant_record in maas_multitenant_upgrade_tenant_records:
            verify_multitenant_upgrade_models_list_probe(
                admin_client=admin_client,
                persist_namespace=persist_namespace,
                maas_host=maas_host,
                request_session_http=request_session_http,
                tenant_record=tenant_record,
            )

    @pytest.mark.parametrize("owner_index", MULTITENANT_UPGRADE_TENANT_OWNER_INDEX)
    @pytest.mark.post_upgrade
    def test_multitenant_inference_and_cross_tenant_auth_post_upgrade(
        self,
        owner_index: int,
        admin_client: DynamicClient,
        maas_multitenant_upgrade_namespace: Namespace,
        maas_host: str,
        request_session_http: requests.Session,
        maas_multitenant_upgrade_tenant_records: list[MultitenantUpgradeTenantRecord],
    ) -> None:
        """Given persisted pre-upgrade API keys, when posting inference on own vs foreign gateway,
        then the owner's gateway returns 200 and the peer gateway returns 401 or 403.
        """
        tenant_a_record, tenant_b_record = _tenant_pair(
            tenant_records=maas_multitenant_upgrade_tenant_records,
        )
        owner_record = tenant_a_record if owner_index == 0 else tenant_b_record
        peer_record = tenant_b_record if owner_index == 0 else tenant_a_record
        persist_namespace = maas_multitenant_upgrade_namespace.name
        chat_payload = multitenant_upgrade_inference_chat_payload()
        assert_multitenant_upgrade_inference_isolation_for_pair(
            admin_client=admin_client,
            persist_namespace=persist_namespace,
            maas_host=maas_host,
            request_session_http=request_session_http,
            owner_tenant_record=owner_record,
            peer_tenant_record=peer_record,
            chat_payload=chat_payload,
        )

    @pytest.mark.post_upgrade
    def test_multitenant_rate_limit_isolation_post_upgrade(
        self,
        admin_client: DynamicClient,
        maas_multitenant_upgrade_namespace: Namespace,
        maas_host: str,
        request_session_http: requests.Session,
        maas_multitenant_upgrade_tenant_records: list[MultitenantUpgradeTenantRecord],
    ) -> None:
        """Given asymmetric subscription token limits, when tenant A is rate-limited post-upgrade,
        then tenant B inference still succeeds (per-tenant TRLP isolation).
        """
        tenant_a_record, tenant_b_record = _tenant_pair(
            tenant_records=maas_multitenant_upgrade_tenant_records,
        )
        persist_namespace = maas_multitenant_upgrade_namespace.name
        assert_multitenant_upgrade_rate_limit_isolation(
            admin_client=admin_client,
            persist_namespace=persist_namespace,
            maas_host=maas_host,
            request_session_http=request_session_http,
            rate_limited_tenant_record=tenant_a_record,
            peer_tenant_record=tenant_b_record,
        )
        LOGGER.info("Multitenant upgrade per-tenant rate limit isolation validated post-upgrade")
