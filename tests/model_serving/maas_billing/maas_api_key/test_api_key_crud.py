from __future__ import annotations

import pytest
import requests
import structlog

from tests.model_serving.maas_billing.maas_api_key.utils import get_api_key, list_api_keys
from tests.model_serving.maas_billing.utils import create_api_key
from utilities.general import generate_random_name

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "maas_api_gateway_reachable",
    "minimal_subscription_for_free_user",
)
class TestAPIKeyCRUD:
    """Tests for MaaS API key lifecycle: create, list, and revoke."""

    @pytest.mark.smoke
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

        for field in ("id", "key", "name"):
            assert field in body, f"Expected '{field}' field in create response"

        key = body["key"]
        assert key.startswith("sk-oai-"), "Expected key to start with 'sk-oai-' prefix"
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
        two_active_api_key_ids: list[str],
    ) -> None:
        """Verify active API keys are listed and no plaintext key is exposed."""

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
        for created_id in two_active_api_key_ids:
            assert created_id in key_ids, f"Created key id={created_id} not found in listed keys"

        for item in items:
            assert "key" not in item, f"Plaintext key must not appear in any list item: {item}"

        LOGGER.info(f"[list] Found {len(items)} active keys")

    @pytest.mark.tier2
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "admin"}], indirect=True)
    def test_list_api_keys_pagination(
        self,
        request_session_http: requests.Session,
        base_url: str,
        ocp_token_for_actor: str,
    ) -> None:
        """Verify that the search endpoint respects the pagination limit."""

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

    @pytest.mark.smoke
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "admin"}], indirect=True)
    def test_revoke_api_key(
        self,
        request_session_http: requests.Session,
        base_url: str,
        ocp_token_for_actor: str,
        revoked_api_key_id: str,
    ) -> None:
        """Verify a revoked API key shows status='revoked' on subsequent GET."""

        get_resp, get_body = get_api_key(
            request_session_http=request_session_http,
            base_url=base_url,
            key_id=revoked_api_key_id,
            ocp_user_token=ocp_token_for_actor,
        )
        assert get_resp.status_code == 200, (
            f"Expected 200 on GET after revoke, got {get_resp.status_code}: {get_resp.text[:200]}"
        )
        assert get_body.get("status") == "revoked", f"Expected status='revoked' on GET after revoke, got: {get_body}"
        LOGGER.info(f"[revoke] Key {revoked_api_key_id} confirmed revoked")
