import pytest
import requests
from ocp_resources.route import Route

from tests.ai_safety.evalhub.constants import (
    EVALHUB_HEALTH_PATH,
    EVALHUB_HEALTH_STATUS_HEALTHY,
    EVALHUB_HEALTHZ_PATH,
)
from tests.ai_safety.evalhub.utils import build_headers, validate_evalhub_health
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
        model_namespace,
    ) -> None:
        """Verify the EvalHub service responds with healthy status."""
        validate_evalhub_health(
            host=evalhub_route.host,
            token=current_client_token,
            ca_bundle_file=evalhub_ca_bundle_file,
            tenant_namespace=model_namespace.name,
        )

    def test_evalhub_health_requires_tenant(
        self,
        current_client_token: str,
        evalhub_ca_bundle_file: str,
        evalhub_route: Route,
    ) -> None:
        """/api/v1/health requires X-Tenant header."""
        url = f"https://{evalhub_route.host}{EVALHUB_HEALTH_PATH}"
        headers = build_headers(token=current_client_token)

        response = requests.get(
            url=url,
            headers=headers,
            verify=evalhub_ca_bundle_file,
            timeout=10,
        )
        assert response.status_code == 400, f"Expected 400 without X-Tenant, got {response.status_code}"

    def test_evalhub_healthz_is_tenant_agnostic(
        self,
        current_client_token: str,
        evalhub_ca_bundle_file: str,
        evalhub_route: Route,
    ) -> None:
        """/healthz works without X-Tenant header.

        /healthz is the kubelet probe endpoint — unauthenticated and
        does not require identity headers, unlike /api/v1/health.
        """
        url = f"https://{evalhub_route.host}{EVALHUB_HEALTHZ_PATH}"
        headers = get_auth_headers(token=current_client_token)

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
        model_namespace,
        method: str,
    ) -> None:
        """Health endpoint rejects POST, PUT, and DELETE with 405."""
        url = f"https://{evalhub_route.host}{EVALHUB_HEALTH_PATH}"
        headers = build_headers(token=current_client_token, tenant=model_namespace.name)
        response = getattr(requests, method)(
            url=url,
            headers=headers,
            verify=evalhub_ca_bundle_file,
            timeout=10,
        )
        assert response.status_code == 405, (
            f"Expected 405 for {method.upper()} on health endpoint, got {response.status_code}"
        )
