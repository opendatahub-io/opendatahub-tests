import pytest
from ocp_resources.route import Route

from tests.model_explainability.evalhub.utils import validate_evalhub_health, validate_evalhub_providers


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-health-providers"},
        ),
    ],
    indirect=True,
)
@pytest.mark.smoke
@pytest.mark.model_explainability
class TestEvalHub:
    """Tests for basic EvalHub service health and providers."""

    def test_evalhub_health_endpoint(
        self,
        current_client_token: str,
        evalhub_ca_bundle_file: str,
        evalhub_route: Route,
    ) -> None:
        """Verify the EvalHub service responds with healthy status via kube-rbac-proxy."""
        validate_evalhub_health(
            host=evalhub_route.host,
            token=current_client_token,
            ca_bundle_file=evalhub_ca_bundle_file,
        )

    def test_evalhub_providers_list(
        self,
        current_client_token: str,
        evalhub_ca_bundle_file: str,
        evalhub_route: Route,
        model_namespace,
    ) -> None:
        """Test that the evaluations providers endpoint returns a non-empty list."""

        validate_evalhub_providers(
            host=evalhub_route.host,
            token=current_client_token,
            ca_bundle_file=evalhub_ca_bundle_file,
            tenant=model_namespace.name,
        )
