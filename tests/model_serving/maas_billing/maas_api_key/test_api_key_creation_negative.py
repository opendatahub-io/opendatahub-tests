from __future__ import annotations

import pytest
import requests
import structlog

from tests.model_serving.maas_billing.maas_api_key.utils import assert_key_rejected_on_endpoint
from tests.model_serving.maas_billing.utils import create_api_key, revoke_api_key
from utilities.general import generate_random_name

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "maas_api_gateway_reachable",
    "maas_model_tinyllama_free",
    "maas_auth_policy_tinyllama_free",
    "maas_subscription_tinyllama_free",
)
class TestApiKeyCreationNegative:
    """Negative tests for API key creation and revoked key on API endpoints."""

    @pytest.mark.tier3
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_create_key_with_empty_name_rejected(
        self,
        request_session_http: requests.Session,
        base_url: str,
        ocp_token_for_actor: str,
    ) -> None:
        """Verify empty key name is rejected with 400."""
        response, _ = create_api_key(
            base_url=base_url,
            ocp_user_token=ocp_token_for_actor,
            request_session_http=request_session_http,
            api_key_name="",
            raise_on_error=False,
        )
        assert response.status_code == 400, (
            f"Expected 400 for empty key name, got {response.status_code}: "
            f"{(response.text or '')[:200]}"
        )
        LOGGER.info(f"Empty key name correctly rejected with {response.status_code}")

    @pytest.mark.tier3
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_create_key_with_nonexistent_subscription_rejected(
        self,
        request_session_http: requests.Session,
        base_url: str,
        ocp_token_for_actor: str,
    ) -> None:
        """Verify non-existent subscription is rejected with 400."""
        key_name = f"e2e-bad-sub-{generate_random_name()}"
        response, _ = create_api_key(
            base_url=base_url,
            ocp_user_token=ocp_token_for_actor,
            request_session_http=request_session_http,
            api_key_name=key_name,
            subscription="nonexistent-subscription-xyz",
            raise_on_error=False,
        )
        assert response.status_code == 400, (
            f"Expected 400 for non-existent subscription, got {response.status_code}: "
            f"{(response.text or '')[:200]}"
        )
        LOGGER.info(f"Non-existent subscription correctly rejected with {response.status_code}")

    @pytest.mark.tier3
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_create_duplicate_key_name_allowed(
        self,
        request_session_http: requests.Session,
        base_url: str,
        ocp_token_for_actor: str,
        maas_subscription_tinyllama_free,
    ) -> None:
        """Verify duplicate key names are allowed since names have no unique constraint."""
        key_name = f"e2e-duplicate-{generate_random_name()}"
        created_key_ids = []

        for attempt in range(2):
            response, body = create_api_key(
                base_url=base_url,
                ocp_user_token=ocp_token_for_actor,
                request_session_http=request_session_http,
                api_key_name=key_name,
                subscription=maas_subscription_tinyllama_free.name,
                raise_on_error=False,
            )
            assert response.status_code in (200, 201), (
                f"Key creation attempt {attempt + 1} failed: {response.status_code}: "
                f"{(response.text or '')[:200]}"
            )
            if body.get("id"):
                created_key_ids.append(body["id"])

        LOGGER.info(f"Duplicate key name correctly allowed: created {len(created_key_ids)} keys")

        for key_id in created_key_ids:
            revoke_api_key(
                request_session_http=request_session_http,
                base_url=base_url,
                key_id=key_id,
                ocp_user_token=ocp_token_for_actor,
            )

    @pytest.mark.tier3
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_revoked_key_rejected_on_models_endpoint(
        self,
        request_session_http: requests.Session,
        base_url: str,
        ocp_token_for_actor: str,
        maas_subscription_tinyllama_free,
    ) -> None:
        """Verify revoked API key is rejected with 403 on /v1/models."""
        key_name = f"e2e-revoke-models-{generate_random_name()}"
        _, body = create_api_key(
            base_url=base_url,
            ocp_user_token=ocp_token_for_actor,
            request_session_http=request_session_http,
            api_key_name=key_name,
            subscription=maas_subscription_tinyllama_free.name,
        )

        revoke_api_key(
            request_session_http=request_session_http,
            base_url=base_url,
            key_id=body["id"],
            ocp_user_token=ocp_token_for_actor,
        )

        assert_key_rejected_on_endpoint(
            request_session_http=request_session_http,
            url=f"{base_url}/v1/models",
            plaintext_key=body["key"],
        )
        LOGGER.info(f"Revoked key correctly rejected on /v1/models with 403")
