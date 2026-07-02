import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from pytest_testconfig import config as py_config

from tests.model_serving.maas_billing.maas_api_key.utils import get_api_key, list_api_keys
from tests.model_serving.maas_billing.multitenancy.aitenant.utils import AITenantTestContext
from tests.model_serving.maas_billing.multitenancy.utils import (
    assert_api_key_search_excludes_ids,
    assert_api_key_search_includes_ids,
    gateway_ref_from_aitenant,
    verify_tenant_gateway_auth_policy_callback_url,
)
from tests.model_serving.maas_billing.utils import assert_api_key_created_ok, create_api_key, revoke_api_key
from utilities.general import generate_random_name

LOGGER = structlog.get_logger(name=__name__)

ACTIVE_KEY_FILTERS = {"status": ["active"]}
ACTIVE_KEY_SORT = {"by": "created_at", "order": "desc"}
ACTIVE_KEY_PAGINATION = {"limit": 50, "offset": 0}


@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "authorino_tls_configured",
    "maas_gateway_api",
    "aitenant_infra_namespace",
    "isolation_tenant_governance",
    "per_tenant_maas_api_ready",
)
class TestMultitenancyAuthIsolation:
    """Verify maas-api API key data is isolated per tenant Gateway."""

    @pytest.mark.tier2
    def test_tenant_gateway_auth_policy_callback_targets_per_tenant_maas_api(
        self,
        admin_client: DynamicClient,
        two_aitenant_test_contexts: tuple[AITenantTestContext, AITenantTestContext],
    ) -> None:
        """Given Ready AITenants with tenant MaaSAuthPolicies, when reading each Gateway MaaS AuthPolicy,
        then the apiKeyValidation callback URL targets maas-api-{tenant} in the applications namespace.
        """
        applications_namespace = py_config["applications_namespace"]
        for test_context in two_aitenant_test_contexts:
            gateway_name, gateway_namespace = gateway_ref_from_aitenant(aitenant=test_context["aitenant"])
            verify_tenant_gateway_auth_policy_callback_url(
                admin_client=admin_client,
                gateway_name=gateway_name,
                gateway_namespace=gateway_namespace,
                aitenant_name=test_context["aitenant_name"],
                applications_namespace=applications_namespace,
            )

    @pytest.mark.tier2
    def test_create_api_key_in_tenant_gateway_succeeds(
        self,
        request_session_http: requests.Session,
        tenant_a_base_url: str,
        tenant_a_subscription_name: str,
        current_client_token: str,
    ) -> None:
        """Given a Ready AITenant with an external Gateway Route, when creating an API key
        through that tenant's maas-api URL, then maas-api returns 201 Created.
        """
        api_key_name = f"e2e-mt-auth-create-{generate_random_name()}"
        create_response, create_body = create_api_key(
            base_url=tenant_a_base_url,
            ocp_user_token=current_client_token,
            request_session_http=request_session_http,
            api_key_name=api_key_name,
            subscription=tenant_a_subscription_name,
        )
        assert_api_key_created_ok(resp=create_response, body=create_body)
        key_id = create_body["id"]
        LOGGER.info(f"[auth-isolation] Created tenant-scoped key id={key_id} at {tenant_a_base_url}")
        revoke_response, _ = revoke_api_key(
            request_session_http=request_session_http,
            base_url=tenant_a_base_url,
            key_id=key_id,
            ocp_user_token=current_client_token,
        )
        assert revoke_response.status_code == 200, (
            f"Expected 200 on cleanup revoke for key id={key_id}, "
            f"got {revoke_response.status_code}: {revoke_response.text[:200]}"
        )

    @pytest.mark.tier2
    def test_get_api_key_from_other_tenant_gateway_not_found(
        self,
        request_session_http: requests.Session,
        tenant_b_base_url: str,
        current_client_token: str,
        tenant_a_api_key_id: str,
    ) -> None:
        """Given an API key created in tenant A, when fetching it via tenant B's maas-api URL,
        then maas-api returns 404 Not Found.
        """
        get_response, _ = get_api_key(
            request_session_http=request_session_http,
            base_url=tenant_b_base_url,
            key_id=tenant_a_api_key_id,
            ocp_user_token=current_client_token,
        )
        assert get_response.status_code == 404, (
            f"Expected 404 when reading tenant A key id={tenant_a_api_key_id} "
            f"via tenant B URL {tenant_b_base_url!r}, "
            f"got {get_response.status_code}: {get_response.text[:200]}"
        )
        LOGGER.info(
            f"[auth-isolation] Cross-tenant GET returned 404 for key id={tenant_a_api_key_id} "
            f"via tenant B URL {tenant_b_base_url!r}"
        )

    @pytest.mark.tier2
    def test_search_in_tenant_does_not_return_other_tenant_keys(
        self,
        request_session_http: requests.Session,
        tenant_a_base_url: str,
        current_client_token: str,
        tenant_a_api_key_id: str,
        tenant_b_api_key_id: str,
    ) -> None:
        """Given active keys in tenants A and B, when searching via tenant A's maas-api URL,
        then results include tenant A keys and exclude tenant B keys.
        """
        list_response, list_body = list_api_keys(
            request_session_http=request_session_http,
            base_url=tenant_a_base_url,
            ocp_user_token=current_client_token,
            filters=ACTIVE_KEY_FILTERS,
            sort=ACTIVE_KEY_SORT,
            pagination=ACTIVE_KEY_PAGINATION,
        )
        assert list_response.status_code == 200, (
            f"Expected 200 on tenant A search, got {list_response.status_code}: {list_response.text[:200]}"
        )
        assert_api_key_search_includes_ids(
            list_body=list_body,
            expected_key_ids={tenant_a_api_key_id},
        )
        assert_api_key_search_excludes_ids(
            list_body=list_body,
            excluded_key_ids={tenant_b_api_key_id},
        )
        LOGGER.info(
            f"[auth-isolation] Tenant A search returned tenant A key id={tenant_a_api_key_id} "
            f"and excluded tenant B key id={tenant_b_api_key_id}"
        )
