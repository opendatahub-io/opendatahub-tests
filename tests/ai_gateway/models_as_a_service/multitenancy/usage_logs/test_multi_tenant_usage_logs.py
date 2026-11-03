"""Multi-tenant usage-log dashboard E2E (RHOAIENG-99404)."""

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient

from tests.ai_gateway.models_as_a_service.multitenancy.utils import assert_tenant_inference_status
from tests.ai_gateway.models_as_a_service.observability.usage_logs_query_utils import (
    resolve_openshift_username,
    wait_for_subscriptions_in_loki,
)
from utilities.logger import RedactedString

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "authorino_tls_configured",
    "maas_gateway_api",
    "aitenant_infra_namespace",
    "isolation_tenant_governance",
    "per_tenant_maas_api_ready",
    "multitenancy_usage_logs_telemetry_enabled",
    "usage_logs_multi_tenant_e2e_monitoring_namespace",
)
class TestMultiTenantUsageLogs:
    """Verify usage logs aggregate across AI tenants for admin Loki and personal tenancy-proxy queries."""

    @pytest.mark.tier2
    def test_usage_logs_visible_across_two_tenant_gateways(
        self,
        admin_client: DynamicClient,
        request_session_http: requests.Session,
        tenant_a_inference_url: str,
        tenant_b_inference_url: str,
        tenant_a_inference_headers: dict[str, str],
        tenant_b_inference_headers: dict[str, str],
        tenant_inference_chat_payload: dict,
        tenant_a_subscription_name: str,
        tenant_b_subscription_name: str,
        current_client_token: RedactedString,
        usage_logs_multi_tenant_e2e_monitoring_namespace: str,
        usage_logs_ingestion_wait_timeout: int,
    ) -> None:
        """Given one user infers on tenant A and tenant B, admin and personal LogQL show both subscriptions.

        Maps to RHOAIENG-99404 acceptance criteria for administrator and personal usage dashboards
        (validated via Loki and usage-logs-tenancy-proxy LogQL, not Perses UI).
        """
        user_token = str(current_client_token)
        user_id = resolve_openshift_username(admin_client=admin_client, bearer_token=user_token)
        expected_subscriptions = {tenant_a_subscription_name, tenant_b_subscription_name}

        assert_tenant_inference_status(
            session=request_session_http,
            inference_url=tenant_a_inference_url,
            headers=tenant_a_inference_headers,
            payload=tenant_inference_chat_payload,
            expected_status=200,
        )
        assert_tenant_inference_status(
            session=request_session_http,
            inference_url=tenant_b_inference_url,
            headers=tenant_b_inference_headers,
            payload=tenant_inference_chat_payload,
            expected_status=200,
        )
        LOGGER.info(
            "Issued inference on both tenant gateways",
            user_id=user_id,
            subscriptions=sorted(expected_subscriptions),
        )

        admin_subscriptions = wait_for_subscriptions_in_loki(
            admin_client=admin_client,
            monitoring_namespace=usage_logs_multi_tenant_e2e_monitoring_namespace,
            user_id=user_id,
            expected_subscriptions=expected_subscriptions,
            query_via_proxy=False,
            user_bearer_token=user_token,
            timeout=usage_logs_ingestion_wait_timeout,
        )
        assert expected_subscriptions.issubset(admin_subscriptions), (
            f"Admin Loki query missing subscriptions: expected {sorted(expected_subscriptions)}, "
            f"found {sorted(admin_subscriptions)}"
        )

        personal_subscriptions = wait_for_subscriptions_in_loki(
            admin_client=admin_client,
            monitoring_namespace=usage_logs_multi_tenant_e2e_monitoring_namespace,
            user_id=user_id,
            expected_subscriptions=expected_subscriptions,
            query_via_proxy=True,
            user_bearer_token=user_token,
            timeout=usage_logs_ingestion_wait_timeout,
        )
        assert expected_subscriptions.issubset(personal_subscriptions), (
            f"Personal tenancy-proxy query missing subscriptions: expected {sorted(expected_subscriptions)}, "
            f"found {sorted(personal_subscriptions)}"
        )
