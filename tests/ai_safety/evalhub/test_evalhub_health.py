import pytest
import requests
from ocp_resources.route import Route

from tests.ai_safety.evalhub.constants import (
    EVALHUB_HEALTH_PATH,
    EVALHUB_HEALTH_STATUS_HEALTHY,
)
from tests.ai_safety.evalhub.utils import validate_evalhub_health
from utilities.guardrails import get_auth_headers


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-health"},
        ),
    ],
    indirect=True,
)
@pytest.mark.tier1
@pytest.mark.ai_safety
class TestEvalHub:
    """Tests for basic EvalHub service health."""

    def test_evalhub_health_endpoint(
        self,
        current_client_token: str,
        evalhub_ca_bundle_file: str,
        evalhub_route: Route,
    ) -> None:
        """Given: a running EvalHub instance.
        When: GET /api/v1/health is called.
        Then: response is 200 with status 'healthy'.
        """
        validate_evalhub_health(
            host=evalhub_route.host,
            token=current_client_token,
            ca_bundle_file=evalhub_ca_bundle_file,
        )

    def test_evalhub_health_is_tenant_agnostic(
        self,
        current_client_token: str,
        evalhub_ca_bundle_file: str,
        evalhub_route: Route,
    ) -> None:
        """Given: a running EvalHub instance.
        When: GET /api/v1/health is called with and without X-Tenant header.
        Then: both responses are 200 with status 'healthy'.
        """
        url = f"https://{evalhub_route.host}{EVALHUB_HEALTH_PATH}"
        headers = get_auth_headers(token=current_client_token)

        # Without X-Tenant — should work
        response = requests.get(
            url=url,
            headers=headers,
            verify=evalhub_ca_bundle_file,
            timeout=10,
        )
        response.raise_for_status()
        assert response.json()["status"] == EVALHUB_HEALTH_STATUS_HEALTHY

        # With X-Tenant — should also work (header ignored)
        headers["X-Tenant"] = "nonexistent-namespace"
        response = requests.get(
            url=url,
            headers=headers,
            verify=evalhub_ca_bundle_file,
            timeout=10,
        )
        response.raise_for_status()
        assert response.json()["status"] == EVALHUB_HEALTH_STATUS_HEALTHY

    @pytest.mark.parametrize("method", ["post", "put", "delete"])
    def test_evalhub_health_rejects_non_get_methods(
        self,
        current_client_token: str,
        evalhub_ca_bundle_file: str,
        evalhub_route: Route,
        method: str,
    ) -> None:
        """Given: a running EvalHub instance.
        When: a non-GET method is sent to /api/v1/health.
        Then: response is 405 (method not allowed).
        """
        url = f"https://{evalhub_route.host}{EVALHUB_HEALTH_PATH}"
        headers = get_auth_headers(token=current_client_token)
        response = getattr(requests, method)(
            url=url,
            headers=headers,
            verify=evalhub_ca_bundle_file,
            timeout=10,
        )
        assert response.status_code == 405, (
            f"Expected 405 for {method.upper()} on health endpoint, got {response.status_code}"
        )
