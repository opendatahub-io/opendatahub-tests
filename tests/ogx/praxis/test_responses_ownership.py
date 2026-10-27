"""Ownership of the `/v1/responses` API after the OGX -> Praxis migration."""

from typing import Final

import httpx
import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route
from ogx_client import OgxClient

from tests.ogx.constants import OGX_CLIENT_VERIFY_SSL, OGX_CORE_INFERENCE_MODEL

NAMESPACE_PARAMS = {"name": "test-praxis-responses-ownership", "randomize_name": True}
RESPONSES_API_PATH: Final[str] = "/v1/responses"
UNSERVED_STATUS_CODES: Final[frozenset[int]] = frozenset({404, 503})


@pytest.mark.parametrize(
    "unprivileged_model_namespace",
    [pytest.param(NAMESPACE_PARAMS)],
    indirect=True,
)
@pytest.mark.ogx
@pytest.mark.skip_must_gather
class TestPraxisResponsesOwnership:
    """Verify Praxis exclusively exposes `/v1/responses`."""

    @pytest.mark.tier1
    def test_responses_served_exclusively_by_praxis(
        self,
        admin_client: DynamicClient,
        unprivileged_model_namespace: Namespace,
        praxis_gateway_route: Route,
        ogx_praxis_client: OgxClient,
    ) -> None:
        """Verify Praxis is the namespace's sole externally routed Responses API.

        Given: A Praxis gateway exposed in a test namespace.
        When: A Responses request is sent to Praxis and all other namespace Routes are probed.
        Then: Praxis serves the request and no other Route exposes `/v1/responses`.
        """
        response = ogx_praxis_client.responses.create(
            model=OGX_CORE_INFERENCE_MODEL,
            input="Reply with the single word 'Praxis'.",
            max_output_tokens=128,
        )
        assert response.output, "Praxis returned no output for the Responses request"

        other_routes = [
            route
            for route in Route.get(client=admin_client, namespace=unprivileged_model_namespace.name)
            if route.name != praxis_gateway_route.name
        ]
        with httpx.Client(verify=OGX_CLIENT_VERIFY_SSL, timeout=10) as probe_client:
            routes_exposing_responses = [
                f"{route.name} (Service {route.instance.to_dict()['spec']['to']['name']})"
                for route in other_routes
                if probe_client.post(f"https://{route.host}{RESPONSES_API_PATH}", json={}).status_code
                not in UNSERVED_STATUS_CODES
            ]

        assert not routes_exposing_responses, (
            f"Routes other than Praxis expose {RESPONSES_API_PATH}: {routes_exposing_responses}"
        )
