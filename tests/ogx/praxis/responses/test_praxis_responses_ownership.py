"""Ownership of the `/v1/responses` API after the OGX -> Praxis migration.

On a greenfield RHOAI 3.6 cluster the Responses API must be served by Praxis
only. These tests derive everything they assert from the cluster itself: the
serving workload is resolved by following the matching HTTPRoute to its backend
Service and from there to its pods, so no Praxis naming convention is hardcoded
and a route pointing at the wrong backend cannot pass unnoticed.
"""

from typing import Final

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.pod import Pod
from ocp_resources.route import Route
from ocp_resources.service import Service

from tests.ogx.constants import OGX_CORE_INFERENCE_MODEL, OGX_CORE_POD_FILTER
from tests.ogx.praxis.constants import PROBE_TIMEOUT_SECONDS, REQUEST_TIMEOUT_SECONDS, RESPONSES_API_PATH
from tests.ogx.praxis.utils import (
    backend_services,
    http_routes_matching_path,
    pod_logs,
    pods_for_service,
    route_url,
)
from utilities.resources.http_route import HTTPRoute

LOGGER = structlog.get_logger(name=__name__)

# HTTP statuses an external endpoint returns when it does not implement a path.
# Anything else means the path is served there, including auth rejections, which
# prove the implementation is reachable.
UNSERVED_STATUS_CODES: Final[frozenset[int]] = frozenset({404, 503})

RESPONSES_PROMPT: Final[str] = "Summarize the Q3 migration report."
RESPONSES_MAX_OUTPUT_TOKENS: Final[int] = 128


def _responses_path_status(session: requests.Session, url: str) -> int | None:
    """Probe whether an endpoint implements the Responses API path.

    The probe is unauthenticated on purpose: an authentication rejection still
    proves the path is implemented and reachable.

    Args:
        session: HTTP session carrying no OGX SDK authentication.
        url: Full URL of the Responses API path to probe.

    Returns:
        The HTTP status code, or None when the endpoint is unreachable.
    """
    try:
        response = session.post(url=url, json={}, timeout=PROBE_TIMEOUT_SECONDS)
    except requests.RequestException as exception:
        LOGGER.info(f"Probe of {url} did not connect: {type(exception).__name__}")
        return None
    return response.status_code


@pytest.fixture
def responses_http_routes(admin_client: DynamicClient) -> list[HTTPRoute]:
    """Every HTTPRoute in the cluster that declares the `/v1/responses` path."""
    return http_routes_matching_path(client=admin_client, path=RESPONSES_API_PATH)


@pytest.fixture
def ogx_backed_routes(admin_client: DynamicClient) -> list[Route]:
    """OpenShift Routes whose target Service selects OGX core pods."""
    ogx_pod_keys = {
        (pod.namespace, pod.name) for pod in Pod.get(client=admin_client, label_selector=OGX_CORE_POD_FILTER)
    }
    if not ogx_pod_keys:
        return []

    matching_routes: list[Route] = []
    for route in Route.get(client=admin_client):
        target = route.instance.to_dict()["spec"].get("to") or {}
        if target.get("kind", "Service") != "Service" or not target.get("name"):
            continue
        service = Service(client=admin_client, name=target["name"], namespace=route.namespace)
        if not service.exists:
            continue
        if any(
            (pod.namespace, pod.name) in ogx_pod_keys for pod in pods_for_service(client=admin_client, service=service)
        ):
            matching_routes.append(route)
    return matching_routes


@pytest.mark.ogx
@pytest.mark.skip_must_gather
class TestPraxisResponsesOwnership:
    """Verify Praxis is the only implementation of `/v1/responses` reachable from outside the cluster."""

    @pytest.mark.tier1
    def test_v1_responses_served_exclusively_by_praxis(
        self,
        admin_client: DynamicClient,
        responses_http_routes: list[HTTPRoute],
        ogx_backed_routes: list[Route],
        request_session: requests.Session,
        tenant_authorization_header: dict[str, str],
    ) -> None:
        """Verify the Responses API is owned by Praxis alone on a greenfield cluster.

        Given a freshly installed cluster exposing the Responses API through the Platform Gateway,
        When an authenticated POST /v1/responses is sent through the external Gateway hostname,
        Then exactly one HTTPRoute declares that path, its backend pods answer the request with a
        valid OpenAI Responses payload, no OGX pod records the request, and no other externally
        exposed Route serves the same path.
        """
        assert len(responses_http_routes) == 1, (
            f"Expected exactly one HTTPRoute declaring {RESPONSES_API_PATH}, found "
            f"{[f'{route.namespace}/{route.name}' for route in responses_http_routes]}"
        )
        responses_route = responses_http_routes[0]

        serving_pods: list[Pod] = [
            pod
            for service in backend_services(client=admin_client, http_route=responses_route, path=RESPONSES_API_PATH)
            if service.exists
            for pod in pods_for_service(client=admin_client, service=service)
        ]

        ogx_pods = list(Pod.get(client=admin_client, label_selector=OGX_CORE_POD_FILTER))
        ogx_pod_keys = {(pod.namespace, pod.name) for pod in ogx_pods}
        serving_ogx_pods = sorted(
            f"{pod.namespace}/{pod.name}" for pod in serving_pods if (pod.namespace, pod.name) in ogx_pod_keys
        )
        assert not serving_ogx_pods, (
            f"{RESPONSES_API_PATH} is backed by OGX pods {serving_ogx_pods}; it must be served by Praxis"
        )

        hostnames = responses_route.instance.to_dict()["spec"].get("hostnames") or []
        assert hostnames, (
            f"HTTPRoute {responses_route.namespace}/{responses_route.name} exposes no hostname, so "
            f"{RESPONSES_API_PATH} is not reachable from outside the cluster"
        )
        responses_url = f"https://{hostnames[0]}{RESPONSES_API_PATH}"
        LOGGER.info(f"POST {responses_url} with model {OGX_CORE_INFERENCE_MODEL}")
        response = request_session.post(
            url=responses_url,
            headers=tenant_authorization_header,
            json={
                "model": OGX_CORE_INFERENCE_MODEL,
                "input": RESPONSES_PROMPT,
                "max_output_tokens": RESPONSES_MAX_OUTPUT_TOKENS,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        assert response.status_code == 200, (
            f"POST {RESPONSES_API_PATH} returned HTTP {response.status_code}: {response.text[:200]}"
        )

        response_body = response.json()
        response_id = response_body.get("id")
        assert response_id and response_body.get("output"), (
            f"Responses payload is missing a non-empty 'id' and 'output': {response_body}"
        )

        praxis_pods_logging_request = sorted(
            f"{pod.namespace}/{pod.name}" for pod in serving_pods if response_id in pod_logs(pod=pod)
        )
        assert praxis_pods_logging_request, (
            f"None of the pods backing {RESPONSES_API_PATH} "
            f"({sorted(f'{pod.namespace}/{pod.name}' for pod in serving_pods)}) "
            f"recorded the request {response_id}"
        )
        LOGGER.info(f"Request served by Praxis pods {praxis_pods_logging_request}")

        ogx_pods_logging_request = sorted(
            f"{pod.namespace}/{pod.name}" for pod in ogx_pods if response_id in pod_logs(pod=pod)
        )
        assert not ogx_pods_logging_request, (
            f"OGX pods {ogx_pods_logging_request} also recorded the request {response_id}"
        )

        externally_served_by_ogx = sorted(
            url
            for url in (route_url(route=route, path=RESPONSES_API_PATH) for route in ogx_backed_routes)
            if (status_code := _responses_path_status(session=request_session, url=url)) is not None
            and status_code not in UNSERVED_STATUS_CODES
        )
        assert not externally_served_by_ogx, (
            f"OGX-backed Routes still expose {RESPONSES_API_PATH}: {externally_served_by_ogx}"
        )
