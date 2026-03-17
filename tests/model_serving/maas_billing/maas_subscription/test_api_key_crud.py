from __future__ import annotations

import pytest
import requests
from simple_logger.logger import get_logger

from tests.model_serving.maas_billing.maas_subscription.utils import (
    create_api_key,
    get_api_key,
    list_api_keys,
    revoke_api_key,
)
from utilities.general import generate_random_name

LOGGER = get_logger(name=__name__)


@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "maas_api_gateway_reachable",
)
class TestAPIKeyCRUD:
    """Tests for MaaS API key lifecycle: create, list, and revoke."""

    @pytest.mark.tier1
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "admin"}], indirect=True)
    def test_create_api_key(
        self,
        request_session_http: requests.Session,
        base_url: str,
        ocp_token_for_actor: str,
    ) -> None:
        """Verify API key creation and show-once behavior."""

        key_name = f"e2e-crud-create-{generate_random_name()}"

        _, body = create_api_key(
            base_url=base_url,
            ocp_user_token=ocp_token_for_actor,
            request_session_http=request_session_http,
            api_key_name=key_name,
        )

        assert "id" in body, f"Expected 'id' in create response, got: {body}"
        assert "key" in body, f"Expected 'key' in create response, got: {body}"
        assert "name" in body, f"Expected 'name' in create response, got: {body}"

        key = body["key"]
        assert key.startswith("sk-oai-"), f"Expected 'sk-oai-' prefix, got: {key[:20]}"
        assert len(key) > len("sk-oai-"), "Key body after prefix must not be empty"

        LOGGER.info(f"[create] Created key id={body['id']}, key_prefix=sk-oai-***")

        get_resp, get_body = get_api_key(
            request_session_http=request_session_http,
            base_url=base_url,
            key_id=body["id"],
            ocp_user_token=ocp_token_for_actor,
        )
        assert get_resp.status_code == 200, (
            f"Expected 200 on GET /v1/api-keys/{body['id']}, got {get_resp.status_code}: {get_resp.text[:200]}"
        )
        assert "key" not in get_body, "Plaintext key must not be returned by GET after creation (show-once pattern)"

    @pytest.mark.tier1
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "admin"}], indirect=True)
    def test_list_api_keys(
        self,
        request_session_http: requests.Session,
        base_url: str,
        ocp_token_for_actor: str,
    ) -> None:
        """Verify active API keys are listed and pagination works."""

        key1_name = f"e2e-crud-list-1-{generate_random_name()}"
        key2_name = f"e2e-crud-list-2-{generate_random_name()}"

        _, key1_body = create_api_key(
            base_url=base_url,
            ocp_user_token=ocp_token_for_actor,
            request_session_http=request_session_http,
            api_key_name=key1_name,
        )
        _, key2_body = create_api_key(
            base_url=base_url,
            ocp_user_token=ocp_token_for_actor,
            request_session_http=request_session_http,
            api_key_name=key2_name,
        )
        key1_id = key1_body["id"]
        key2_id = key2_body["id"]

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

        items: list[dict] = list_body.get("items") or list_body.get("data") or []
        assert len(items) >= 2, f"Expected at least 2 active keys, got {len(items)}"

        key_ids = [item["id"] for item in items]
        assert key1_id in key_ids, f"key1 id={key1_id} not found in listed keys: {key_ids}"
        assert key2_id in key_ids, f"key2 id={key2_id} not found in listed keys: {key_ids}"

        for item in items:
            assert "key" not in item, f"Plaintext key must not appear in any list item: {item}"

        LOGGER.info(f"[list] Found {len(items)} active keys")

        page_resp, page_body = list_api_keys(
            request_session_http=request_session_http,
            base_url=base_url,
            ocp_user_token=ocp_token_for_actor,
            filters={"status": ["active"]},
            sort={"by": "created_at", "order": "desc"},
            pagination={"limit": 1, "offset": 0},
        )
        assert page_resp.status_code == 200, (
            f"Expected 200 on paginated search, got {page_resp.status_code}: {page_resp.text[:200]}"
        )
        paged_items: list[dict] = page_body.get("items") or page_body.get("data") or []
        assert len(paged_items) <= 1, f"Expected at most 1 item with limit=1, got {len(paged_items)}"
        LOGGER.info(f"[list] Pagination limit=1 returned {len(paged_items)} item(s)")

    @pytest.mark.tier1
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "admin"}], indirect=True)
    def test_revoke_api_key(
        self,
        request_session_http: requests.Session,
        base_url: str,
        ocp_token_for_actor: str,
    ) -> None:
        """Verify an API key can be revoked and remains revoked on GET."""

        key_name = f"e2e-crud-revoke-{generate_random_name()}"

        _, body = create_api_key(
            base_url=base_url,
            ocp_user_token=ocp_token_for_actor,
            request_session_http=request_session_http,
            api_key_name=key_name,
        )
        key_id = body["id"]

        revoke_resp, revoke_body = revoke_api_key(
            request_session_http=request_session_http,
            base_url=base_url,
            key_id=key_id,
            ocp_user_token=ocp_token_for_actor,
        )
        assert revoke_resp.status_code == 200, (
            f"Expected 200 on DELETE /v1/api-keys/{key_id}, got {revoke_resp.status_code}: {revoke_resp.text[:200]}"
        )
        assert revoke_body.get("status") == "revoked", (
            f"Expected status='revoked' in DELETE response, got: {revoke_body}"
        )

        get_resp, get_body = get_api_key(
            request_session_http=request_session_http,
            base_url=base_url,
            key_id=key_id,
            ocp_user_token=ocp_token_for_actor,
        )
        assert get_resp.status_code == 200, (
            f"Expected 200 on GET after revoke, got {get_resp.status_code}: {get_resp.text[:200]}"
        )
        assert get_body.get("status") == "revoked", f"Expected status='revoked' on GET after revoke, got: {get_body}"
        LOGGER.info(f"[revoke] Key {key_id} confirmed revoked")
