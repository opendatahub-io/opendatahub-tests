import pytest
import requests
from ocp_resources.route import Route

from tests.model_explainability.evalhub.constants import (
    EVALHUB_HEALTH_PATH,
    EVALHUB_HEALTH_STATUS_HEALTHY,
)
from tests.model_explainability.evalhub.utils import validate_evalhub_health
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
@pytest.mark.model_explainability
class TestEvalHub:
    """Tests for basic EvalHub service health."""

    def test_evalhub_health_endpoint(
        self,
        current_client_token: str,
        evalhub_ca_bundle_file: str,
        evalhub_route: Route,
    ) -> None:
        """Verify the EvalHub service responds with healthy status."""
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
        """Health endpoint works without X-Tenant header.

        The health endpoint is not in auth.yaml, so it should not
        require tenant authorization. It should also tolerate an
        X-Tenant header being present (ignored, not rejected).
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
        """Health endpoint rejects POST, PUT, and DELETE with 405."""
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
