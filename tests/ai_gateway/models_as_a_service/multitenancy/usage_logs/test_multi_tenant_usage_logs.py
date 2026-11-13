"""Multi-tenant usage-log dashboard E2E (RHOAIENG-99404).

Tenant models use OCI TinyLlama (``tenant_isolation_model_storage_uri`` in ``conftest.py``), so
local runs do not need AWS credentials for ``ods-ci-wisdom``. Jenkins multitenancy jobs that
use S3 keep the default ``ModelStorage.S3.TINYLLAMA`` fixture in ``multitenancy/conftest.py``.
"""

import portforward
import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient

from tests.ai_gateway.models_as_a_service.multitenancy.utils import assert_tenant_inference_status
from tests.ai_gateway.models_as_a_service.observability.constants import (
    LOKI_GATEWAY_SERVICE_NAME,
    LOKI_GATEWAY_SERVICE_PORT,
    LOKI_QUERY_LOCAL_PORT,
    USAGE_LOGS_TENANCY_PROXY_DEPLOYMENT,
    USAGE_LOGS_TENANCY_PROXY_LOCAL_PORT,
    USAGE_LOGS_TENANCY_PROXY_PORT,
)
from tests.ai_gateway.models_as_a_service.multitenancy.usage_logs.utils import (
    assert_expected_subscriptions_in_loki,
    resolve_openshift_username,
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
        monitoring_namespace = usage_logs_multi_tenant_e2e_monitoring_namespace

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

        with (
            portforward.forward(
                pod_or_service=LOKI_GATEWAY_SERVICE_NAME,
                namespace=monitoring_namespace,
                from_port=LOKI_QUERY_LOCAL_PORT,
                to_port=LOKI_GATEWAY_SERVICE_PORT,
                waiting=20,
            ),
            portforward.forward(
                pod_or_service=USAGE_LOGS_TENANCY_PROXY_DEPLOYMENT,
                namespace=monitoring_namespace,
                from_port=USAGE_LOGS_TENANCY_PROXY_LOCAL_PORT,
                to_port=USAGE_LOGS_TENANCY_PROXY_PORT,
                waiting=20,
            ),
        ):
            assert_expected_subscriptions_in_loki(
                http_session=request_session_http,
                admin_token=user_token,
                user_id=user_id,
                expected_subscriptions=expected_subscriptions,
                query_via_proxy=False,
                user_bearer_token=user_token,
                timeout=usage_logs_ingestion_wait_timeout,
                query_description="Admin Loki",
            )
            assert_expected_subscriptions_in_loki(
                http_session=request_session_http,
                admin_token=user_token,
                user_id=user_id,
                expected_subscriptions=expected_subscriptions,
                query_via_proxy=True,
                user_bearer_token=user_token,
                timeout=usage_logs_ingestion_wait_timeout,
                query_description="Personal tenancy-proxy",
            )
