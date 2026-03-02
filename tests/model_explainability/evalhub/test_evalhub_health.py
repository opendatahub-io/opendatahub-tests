import pytest
from ocp_resources.route import Route

from tests.model_explainability.evalhub.utils import validate_evalhub_health


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-health"},
        ),
    ],
    indirect=True,
)
@pytest.mark.smoke
@pytest.mark.model_explainability
class TestEvalHubHealth:
    """Tests for basic EvalHub service health and availability."""

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
