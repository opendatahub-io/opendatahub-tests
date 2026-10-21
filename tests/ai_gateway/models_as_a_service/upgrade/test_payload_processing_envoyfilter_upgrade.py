"""Payload-processing EnvoyFilter upgrade tests (RHOAIENG-100390)."""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.gateway_gateway_networking_k8s_io import Gateway

from tests.ai_gateway.models_as_a_service.body_base_routing.pre_auth_model_header.utils import (
    verify_bbr_envoy_filter_has_pre_and_post_auth_stages,
)
from tests.ai_gateway.models_as_a_service.upgrade.utils import (
    MultitenantUpgradeTenantRecord,
    aitenant_for_multitenant_upgrade_record,
    verify_legacy_default_maas_ipp_envoyfilter_post_upgrade,
    verify_legacy_default_maas_ipp_payload_processing_ready,
    verify_praxis_aitenant_without_maas_legacy_ipp_envoyfilter_post_upgrade,
)
from utilities.constants import MAAS_GATEWAY_NAME, MAAS_GATEWAY_NAMESPACE


@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "authorino_tls_configured",
    "maas_gateway_api",
    "require_legacy_default_maas_ipp",
)
class TestPreUpgradeLegacyIppEnvoyfilterUpgrade:
    """Baseline legacy IPP EnvoyFilter and payload-processing before cluster upgrade."""

    @pytest.mark.pre_upgrade
    def test_legacy_default_maas_ipp_envoyfilter_present_envoyfilter_upgrade_pre_upgrade(
        self,
        maas_legacy_ipp_envoyfilter_config_patches: list[object],
    ) -> None:
        """Given the default MaaS gateway, when running pre-upgrade checks,
        then the legacy IPP EnvoyFilter exposes pre-auth and post-auth stages.
        """
        verify_bbr_envoy_filter_has_pre_and_post_auth_stages(
            config_patches=maas_legacy_ipp_envoyfilter_config_patches,
            gateway_namespace=MAAS_GATEWAY_NAMESPACE,
        )

    @pytest.mark.pre_upgrade
    def test_legacy_default_maas_ipp_payload_processing_ready_envoyfilter_upgrade_pre_upgrade(
        self,
        admin_client: DynamicClient,
    ) -> None:
        """Given the default MaaS tenant legacy IPP stack, when running pre-upgrade checks,
        then payload-processing Deployment and Service are Ready in the Gateway namespace.
        """
        verify_legacy_default_maas_ipp_payload_processing_ready(
            admin_client=admin_client,
            gateway_namespace=MAAS_GATEWAY_NAMESPACE,
        )


@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "authorino_tls_configured",
    "maas_gateway_api",
    "require_legacy_default_maas_ipp",
)
class TestPostUpgradeLegacyIppEnvoyfilterUpgrade:
    """Validate maas legacy IPP EnvoyFilter attachment after upgrade (default tenant / Gateway namespace)."""

    @pytest.mark.post_upgrade
    def test_legacy_default_maas_ipp_envoyfilter_attachment_envoyfilter_upgrade_post_upgrade(
        self,
        admin_client: DynamicClient,
        maas_upgrade_gateway: Gateway,
    ) -> None:
        """Given upgrade from N-1, when legacy IPP EnvoyFilter is reconciled post-upgrade,
        then attachment uses workloadSelector for the default gateway without targetRefs.
        """
        verify_legacy_default_maas_ipp_envoyfilter_post_upgrade(
            admin_client=admin_client,
            gateway_namespace=MAAS_GATEWAY_NAMESPACE,
            gateway_name=maas_upgrade_gateway.name or MAAS_GATEWAY_NAME,
        )


@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "authorino_tls_configured",
    "maas_gateway_api",
    "aitenant_infra_namespace",
    "maas_multitenant_upgrade_namespace",
    "maas_multitenant_upgrade_tenant_records",
)
class TestPostUpgradePraxisAitenantEnvoyfilterUpgrade:
    """Validate Praxis AITenants keep Praxis IPP without maas-controller legacy IPP markers post-upgrade."""

    @pytest.mark.post_upgrade
    def test_praxis_aitenant_without_maas_legacy_ipp_envoyfilter_envoyfilter_upgrade_post_upgrade(
        self,
        admin_client: DynamicClient,
        aitenant_infra_namespace: str,
        maas_multitenant_upgrade_tenant_records: list[MultitenantUpgradeTenantRecord],
    ) -> None:
        """Given multitenant Praxis AITenants from upgrade baseline, when upgrade completes,
        then each tenant has no maas-controller legacy IPP markers and Praxis payload processing is active.
        """
        for tenant_record in maas_multitenant_upgrade_tenant_records:
            aitenant = aitenant_for_multitenant_upgrade_record(
                admin_client=admin_client,
                aitenant_infra_namespace=aitenant_infra_namespace,
                tenant_record=tenant_record,
            )
            verify_praxis_aitenant_without_maas_legacy_ipp_envoyfilter_post_upgrade(
                admin_client=admin_client,
                aitenant=aitenant,
            )
