from __future__ import annotations

from typing import Any

import pytest
import requests
import structlog

from tests.model_serving.maas_billing.maas_api_key.utils import (
    assert_tenant_field_empty,
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
    """Tests verifying the tenant field is present and defaults to empty string in API key responses."""

    @pytest.mark.smoke
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "admin"}], indirect=True)
    def test_create_api_key_response_includes_tenant_field(
        self,
        active_api_key_with_plaintext: dict[str, Any],
    ) -> None:
        """Verify POST /v1/api-keys response includes tenant field defaulting to empty string."""
        assert_tenant_field_empty(body=active_api_key_with_plaintext, context="POST /v1/api-keys")
        LOGGER.info(f"[tenant] Create response includes tenant='' for key id={active_api_key_with_plaintext['id']}")

    @pytest.mark.tier1
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "admin"}], indirect=True)
    def test_get_api_key_response_includes_tenant_field(
        self,
        request_session_http: requests.Session,
        base_url: str,
        ocp_token_for_actor: str,
        active_api_key_id: str,
    ) -> None:
        """Verify GET /v1/api-keys/{id} response includes tenant field defaulting to empty string."""

        get_resp, get_body = get_api_key(
            request_session_http=request_session_http,
            base_url=base_url,
            key_id=active_api_key_id,
            ocp_user_token=ocp_token_for_actor,
        )
        assert get_resp.status_code == 200, (
            f"Expected 200 on GET /v1/api-keys/{active_api_key_id}, got {get_resp.status_code}: {get_resp.text[:200]}"
        )
        assert_tenant_field_empty(body=get_body, context=f"GET /v1/api-keys/{active_api_key_id}")

        LOGGER.info(f"[tenant] GET response includes tenant='' for key id={active_api_key_id}")

    @pytest.mark.tier1
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "admin"}], indirect=True)
    def test_list_api_keys_all_items_include_tenant_field(
        self,
        request_session_http: requests.Session,
        base_url: str,
        ocp_token_for_actor: str,
        two_active_api_key_ids: list[str],
    ) -> None:
        """Verify POST /v1/api-keys/search items all include tenant field defaulting to empty string."""

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
        assert len(items) == len(two_active_api_key_ids), (
            f"Expected {len(two_active_api_key_ids)} active keys, got {len(items)}"
        )

        for item in items:
            assert_tenant_field_empty(body=item, context=f"search item id={item['id']}")

        LOGGER.info(f"[tenant] All {len(items)} listed keys include tenant=''")
