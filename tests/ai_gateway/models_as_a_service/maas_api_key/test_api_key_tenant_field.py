from __future__ import annotations

import pytest
import requests
import structlog

from tests.ai_gateway.models_as_a_service.maas_api_key.utils import (
    assert_tenant_field,
    get_api_key,
    list_api_keys,
)

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "maas_api_gateway_reachable",
    "minimal_subscription_for_free_user",
)
class TestAPIKeyTenantField:
    """Tests verifying the tenant field is present and defaults to models-as-a-service."""

    @pytest.mark.tier1
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "admin"}], indirect=True)
    def test_get_api_key_response_includes_tenant_field(
        self,
        request_session_http: requests.Session,
        base_url: str,
        ocp_token_for_actor: str,
        active_api_key_id: str,
    ) -> None:
        """Verify GET /v1/api-keys/{id} response includes tenant defaulting to models-as-a-service."""

        get_resp, get_body = get_api_key(
            request_session_http=request_session_http,
            base_url=base_url,
            key_id=active_api_key_id,
            ocp_user_token=ocp_token_for_actor,
        )
        assert get_resp.status_code == 200, (
            f"Expected 200 on GET /v1/api-keys/{active_api_key_id}, got {get_resp.status_code}: {get_resp.text[:200]}"
        )
        assert_tenant_field(body=get_body, context=f"GET /v1/api-keys/{active_api_key_id}")

        LOGGER.info(f"[tenant] GET response includes tenant='models-as-a-service' for key id={active_api_key_id}")

    @pytest.mark.tier1
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "admin"}], indirect=True)
    def test_list_api_keys_all_items_include_tenant_field(
        self,
        request_session_http: requests.Session,
        base_url: str,
        ocp_token_for_actor: str,
        two_active_api_key_ids: list[str],
    ) -> None:
        """Verify POST /v1/api-keys/search items for fixture keys include tenant models-as-a-service."""

        list_resp, list_body = list_api_keys(
            request_session_http=request_session_http,
            base_url=base_url,
            ocp_user_token=ocp_token_for_actor,
            filters={"status": ["active"]},
            sort={"by": "created_at", "order": "desc"},
            pagination={"limit": 50, "offset": 0},
        )
        assert list_resp.status_code == 200, (
            f"Expected 200 on POST /v1/api-keys/search, got {list_resp.status_code}: {list_resp.text[:200]}"
        )

        if "items" in list_body:
            items: list[dict] = list_body["items"]
        elif "data" in list_body:
            items = list_body["data"]
        else:
            raise AssertionError(
                f"Expected 'items' or 'data' key in search response, got keys: {list(list_body.keys())}"
            )

        items_by_id = {item["id"]: item for item in items}
        for key_id in two_active_api_key_ids:
            assert key_id in items_by_id, f"Expected key id={key_id} in search results"
            assert_tenant_field(body=items_by_id[key_id], context=f"search item id={key_id}")

        LOGGER.info(f"[tenant] All {len(two_active_api_key_ids)} fixture keys include tenant='models-as-a-service'")
