from __future__ import annotations

import pytest
import requests
import structlog
from timeout_sampler import TimeoutSampler

from tests.model_serving.maas_billing.maas_subscription.utils import (
    assert_model_info_schema,
    assert_models_response_for_subscription,
)
from tests.model_serving.maas_billing.utils import build_maas_headers

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures(
    "maas_unprivileged_model_namespace",
    "maas_subscription_controller_enabled_latest",
    "maas_gateway_api",
    "maas_api_gateway_reachable",
    "maas_model_tinyllama_free",
    "maas_auth_policy_tinyllama_free",
    "maas_subscription_tinyllama_free",
    "maas_model_tinyllama_premium",
    "maas_auth_policy_tinyllama_premium",
    "maas_subscription_tinyllama_premium",
)
class TestListModels:
    """E2E tests for GET /v1/models subscription-aware model filtering."""

    @pytest.mark.tier1
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_api_key_scoped_to_subscription(
        self,
        request_session_http: requests.Session,
        models_url: str,
        api_key_bound_to_free_subscription: str,
        maas_subscription_tinyllama_free,
    ) -> None:
        """Verify API key returns only models from its bound subscription."""
        headers = build_maas_headers(token=api_key_bound_to_free_subscription)

        response = request_session_http.get(url=models_url, headers=headers, timeout=30)

        expected_sub = maas_subscription_tinyllama_free.name
        models = assert_models_response_for_subscription(
            response=response,
            expected_subscription_name=expected_sub,
        )

        model_ids = [model_entry["id"] for model_entry in models]
        LOGGER.info(f"[models] API key scoped to '{expected_sub}' -> {len(models)} model(s): {model_ids}")

    @pytest.mark.tier1
    def test_unauthenticated_request_401(
        self,
        request_session_http: requests.Session,
        models_url: str,
    ) -> None:
        """Verify request without Authorization header returns 401."""
        response = request_session_http.get(url=models_url, timeout=30)

        assert response.status_code == 401, f"Expected 401, got {response.status_code}: {(response.text or '')[:200]}"
        LOGGER.info(f"[models] GET /v1/models (no auth) -> {response.status_code}")

    @pytest.mark.tier1
    def test_api_key_with_deleted_subscription_403(
        self,
        request_session_http: requests.Session,
        models_url: str,
        api_key_for_deleted_subscription: str,
    ) -> None:
        """Verify API key bound to a deleted subscription returns 403 permission_error."""
        headers = build_maas_headers(token=api_key_for_deleted_subscription)

        last_status: int | None = None
        for response in TimeoutSampler(
            wait_timeout=60,
            sleep=5,
            func=request_session_http.get,
            url=models_url,
            headers=headers,
            timeout=30,
        ):
            last_status = response.status_code
            LOGGER.info(f"[models] Polling deleted subscription -> {last_status}")
            if last_status == 403:
                break

        assert last_status == 403, (
            f"Expected 403 for deleted subscription, got {last_status}: {(response.text or '')[:200]}"
        )

        assert "application/json" in response.headers.get("Content-Type", ""), (
            f"Expected JSON response for 403, got Content-Type: {response.headers.get('Content-Type')}"
        )
        error_body = response.json()
        assert "error" in error_body, "Response missing 'error' field"
        assert error_body["error"].get("type") == "permission_error", (
            f"Expected error type 'permission_error', got {error_body['error'].get('type')}"
        )
        LOGGER.info(f"[models] API key with deleted subscription -> {last_status} (permission_error)")

    @pytest.mark.tier2
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_subscription_without_auth_policy_still_lists_model(
        self,
        request_session_http: requests.Session,
        models_url: str,
        ocp_token_for_actor: str,
        free_actor_premium_subscription,
    ) -> None:
        """Verify /v1/models returns models based on subscription ownership, not AuthPolicy."""
        headers = build_maas_headers(token=ocp_token_for_actor)
        headers["x-maas-subscription"] = free_actor_premium_subscription.name

        response = request_session_http.get(url=models_url, headers=headers, timeout=30)

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {(response.text or '')[:200]}"

        data = response.json()
        assert "data" in data, "Response missing 'data' field"

        models = data["data"]
        assert models is not None, "'data' must not be null"
        assert isinstance(models, list), f"'data' must be a list, got {type(models).__name__}"
        assert len(models) == 1, (
            f"Expected 1 model from subscription (visibility granted by subscription ownership), "
            f"got {len(models)}: {models}"
        )
        LOGGER.info(
            f"[models] Subscription without AuthPolicy -> {response.status_code} with {len(models)} model(s) visible"
        )

    @pytest.mark.tier2
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_response_schema_matches_openapi(
        self,
        request_session_http: requests.Session,
        models_url: str,
        api_key_bound_to_free_subscription: str,
    ) -> None:
        """Verify each model entry conforms to the expected ModelInfo schema."""
        headers = build_maas_headers(token=api_key_bound_to_free_subscription)

        response = request_session_http.get(url=models_url, headers=headers, timeout=30)

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {(response.text or '')[:200]}"

        data = response.json()
        assert data.get("object") == "list", f"Expected object='list', got {data.get('object')}"
        assert "data" in data, "Response missing 'data' field"
        assert data["data"] is not None, "'data' field must not be null"

        models = data["data"]
        assert isinstance(models, list), f"'data' must be a list, got {type(models).__name__}"
        assert len(models) > 0, "Expected at least one model for schema validation"

        for model_entry in models:
            assert_model_info_schema(model=model_entry)

        LOGGER.info(f"[models] Response schema validated for {len(models)} model(s)")

    @pytest.mark.tier1
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_user_token_returns_all_accessible_models(
        self,
        request_session_http: requests.Session,
        models_url: str,
        ocp_token_for_actor: str,
    ) -> None:
        """Verify OCP token without subscription header returns models from all accessible subscriptions."""
        headers = build_maas_headers(token=ocp_token_for_actor)
        response = request_session_http.get(url=models_url, headers=headers, timeout=30)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {(response.text or '')[:200]}"
        data = response.json()
        models = data.get("data") or []
        assert len(models) >= 1, f"Expected at least 1 model, got {len(models)}"
        for model_entry in models:
            assert "subscriptions" in model_entry, f"Model '{model_entry.get('id')}' missing 'subscriptions' field"
            assert len(model_entry["subscriptions"]) >= 1, (
                f"Model '{model_entry.get('id')}' should have at least one subscription"
            )
        model_ids = [m["id"] for m in models]
        LOGGER.info(f"[models] User token (no header) -> {len(models)} model(s): {model_ids}")

    @pytest.mark.tier1
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_user_token_with_subscription_header_filters(
        self,
        request_session_http: requests.Session,
        models_url: str,
        ocp_token_for_actor: str,
        maas_subscription_tinyllama_free,
    ) -> None:
        """Verify OCP token with X-MaaS-Subscription header filters to that subscription."""
        headers = build_maas_headers(token=ocp_token_for_actor)
        headers["x-maas-subscription"] = maas_subscription_tinyllama_free.name
        response = request_session_http.get(url=models_url, headers=headers, timeout=30)

        expected_sub = maas_subscription_tinyllama_free.name
        models = assert_models_response_for_subscription(
            response=response,
            expected_subscription_name=expected_sub,
        )
        LOGGER.info(f"[models] OCP token + header '{expected_sub}' -> {len(models)} model(s)")

    @pytest.mark.tier1
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_invalid_subscription_header_403(
        self,
        request_session_http: requests.Session,
        models_url: str,
        ocp_token_for_actor: str,
    ) -> None:
        """Verify non-existent subscription in header returns 403 permission_error."""
        headers = build_maas_headers(token=ocp_token_for_actor)
        headers["x-maas-subscription"] = "nonexistent-subscription-xyz"
        response = request_session_http.get(url=models_url, headers=headers, timeout=30)
        assert response.status_code == 403, (
            f"Expected 403 for invalid subscription, got {response.status_code}: {(response.text or '')[:200]}"
        )
        assert "application/json" in response.headers.get("Content-Type", ""), (
            f"Expected JSON response for 403, got Content-Type: {response.headers.get('Content-Type')}"
        )
        error_body = response.json()
        assert "error" in error_body, "Response missing 'error' field"
        assert error_body["error"].get("type") == "permission_error", (
            f"Expected error type 'permission_error', got {error_body['error'].get('type')}"
        )
        LOGGER.info(f"[models] Invalid subscription header -> {response.status_code} (permission_error)")
