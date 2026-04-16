from __future__ import annotations

import pytest
import requests
import structlog
from ocp_resources.maas_subscription import MaaSSubscription

from tests.model_serving.maas_billing.maas_subscription.utils import assert_models_belong_to_subscription
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
class TestModelsEndpointAccessControl:
    """Security, RBAC, auto-selection, and rate-limit exemption tests for GET /v1/models."""

    @pytest.mark.tier1
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_api_key_ignores_subscription_header(
        self,
        request_session_http: requests.Session,
        models_url: str,
        api_key_bound_to_free_subscription: str,
        maas_subscription_tinyllama_free: MaaSSubscription,
        maas_subscription_tinyllama_premium: MaaSSubscription,
    ) -> None:
        """Verify API key ignores client-injected X-MaaS-Subscription header."""
        headers = build_maas_headers(token=api_key_bound_to_free_subscription)
        headers["x-maas-subscription"] = maas_subscription_tinyllama_premium.name

        response = request_session_http.get(url=models_url, headers=headers, timeout=30)

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {(response.text or '')[:200]}"

        data = response.json()
        models = data.get("data") or []
        assert len(models) >= 1, f"Expected at least 1 model, got {len(models)}"

        expected_sub = maas_subscription_tinyllama_free.name
        assert_models_belong_to_subscription(
            models=models,
            expected_subscription_name=expected_sub,
        )

        LOGGER.info(
            f"[models] API key ignored X-MaaS-Subscription header — "
            f"returned {len(models)} model(s) from bound subscription '{expected_sub}'"
        )

    @pytest.mark.tier1
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_access_denied_to_subscription_403(
        self,
        request_session_http: requests.Session,
        models_url: str,
        ocp_token_for_actor: str,
        maas_subscription_tinyllama_premium: MaaSSubscription,
    ) -> None:
        """Verify user token with a subscription they don't belong to gets 403."""
        headers = build_maas_headers(token=ocp_token_for_actor)
        headers["x-maas-subscription"] = maas_subscription_tinyllama_premium.name

        response = request_session_http.get(url=models_url, headers=headers, timeout=30)

        assert response.status_code == 403, (
            f"Expected 403 for inaccessible subscription, got {response.status_code}: {(response.text or '')[:200]}"
        )

        assert "application/json" in response.headers.get("Content-Type", ""), (
            f"Expected JSON response for 403, got Content-Type: {response.headers.get('Content-Type')}"
        )
        error_body = response.json()
        assert "error" in error_body, "Response missing 'error' field"
        assert error_body["error"].get("type") == "permission_error", (
            f"Expected error type 'permission_error', got {error_body['error'].get('type')}"
        )
        LOGGER.info(
            f"[models] Access denied to subscription '{maas_subscription_tinyllama_premium.name}' "
            f"-> {response.status_code} (permission_error)"
        )

    @pytest.mark.tier1
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_single_subscription_auto_select(
        self,
        request_session_http: requests.Session,
        models_url: str,
        api_key_bound_to_free_subscription: str,
        maas_subscription_tinyllama_free: MaaSSubscription,
    ) -> None:
        """Verify single subscription auto-selects without needing a header."""
        headers = build_maas_headers(token=api_key_bound_to_free_subscription)

        response = request_session_http.get(url=models_url, headers=headers, timeout=30)

        assert response.status_code == 200, (
            f"Expected 200 for single subscription auto-select, got {response.status_code}: "
            f"{(response.text or '')[:200]}"
        )

        data = response.json()
        assert data.get("object") == "list", f"Expected object='list', got {data.get('object')}"
        assert "data" in data, "Response missing 'data' field"

        models = data.get("data") or []
        assert len(models) >= 1, f"Expected at least 1 model from auto-selected subscription, got {len(models)}"

        expected_sub = maas_subscription_tinyllama_free.name
        assert_models_belong_to_subscription(
            models=models,
            expected_subscription_name=expected_sub,
        )

        LOGGER.info(
            f"[models] Single subscription auto-select -> {response.status_code} "
            f"with {len(models)} model(s) from '{expected_sub}'"
        )

    @pytest.mark.tier2
    @pytest.mark.parametrize("ocp_token_for_actor", [{"type": "free"}], indirect=True)
    def test_central_models_endpoint_exempt_from_rate_limiting(
        self,
        request_session_http: requests.Session,
        models_url: str,
        model_url_tinyllama_free: str,
        api_key_bound_to_free_subscription: str,
        maas_subscription_tinyllama_free: MaaSSubscription,
    ) -> None:
        """Verify /v1/models remains accessible when token quota is exhausted."""
        inference_headers = build_maas_headers(token=api_key_bound_to_free_subscription)
        inference_headers["x-maas-subscription"] = maas_subscription_tinyllama_free.name

        max_requests = 20
        rate_limited = False
        for attempt in range(max_requests):
            inference_response = request_session_http.post(
                url=model_url_tinyllama_free,
                headers=inference_headers,
                json={
                    "model": "llm-s3-tinyllama-free",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 50,
                },
                timeout=60,
            )
            if inference_response.status_code == 429:
                rate_limited = True
                LOGGER.info(f"[models] Rate limit hit after {attempt + 1} inference request(s)")
                break

            assert inference_response.status_code == 200, (
                f"Unexpected status {inference_response.status_code} during inference "
                f"(attempt {attempt + 1}): {(inference_response.text or '')[:200]}"
            )

        assert rate_limited, f"Could not exhaust token quota within {max_requests} requests — rate limit not triggered"

        models_headers = build_maas_headers(token=api_key_bound_to_free_subscription)
        models_response = request_session_http.get(
            url=models_url,
            headers=models_headers,
            timeout=30,
        )

        assert models_response.status_code == 200, (
            f"Expected 200 for /v1/models even when quota exhausted, "
            f"got {models_response.status_code}: {(models_response.text or '')[:200]}"
        )

        data = models_response.json()
        assert "data" in data, "Response missing 'data' field"
        assert isinstance(data["data"], list), "'data' must be a list"

        LOGGER.info(
            f"[models] /v1/models exempt from rate limiting -> {models_response.status_code} "
            f"with {len(data['data'])} model(s) (inference blocked with 429)"
        )
