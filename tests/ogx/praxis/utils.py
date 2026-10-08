"""Helpers shared by the OGX -> Praxis migration tests.

The Gateway helpers resolve what actually serves an API path from the cluster
itself: a path is matched against the HTTPRoutes that declare it. Nothing about
Praxis naming is hardcoded, so a missing or ambiguous route cannot pass unnoticed.
"""

from collections.abc import Iterable

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.pod import Pod
from ocp_resources.route import Route
from ocp_resources.service import Service

from utilities.exceptions import UnexpectedResourceCountError
from utilities.resources.http_route import HTTPRoute

LOGGER = structlog.get_logger(name=__name__)


def http_routes_matching_path(client: DynamicClient, path: str) -> list[HTTPRoute]:
    """Return every HTTPRoute in the cluster that explicitly declares `path`.

    Catch-all routes (for example a `PathPrefix: /` rule) are deliberately not
    treated as matches; the contract under test is about the route that owns the
    exact path.

    Args:
        client: Client with cluster-wide read access.
        path: Absolute request path, as declared by a route rule match.

    Returns:
        The routes declaring an exact match on `path`.
    """
    return [
        http_route
        for http_route in HTTPRoute.get(client=client)
        if any(
            (match.get("path") or {}).get("value") == path
            for rule in http_route.instance.to_dict()["spec"].get("rules") or []
            for match in rule.get("matches") or []
        )
    ]


def backend_services(client: DynamicClient, http_route: HTTPRoute) -> list[Service]:
    """Resolve the Service backends an HTTPRoute forwards to.

    Args:
        client: Client with cluster-wide read access.
        http_route: Route whose `backendRefs` are resolved.

    Returns:
        One Service per distinct namespace/name backend reference.
    """
    resolved: dict[tuple[str, str], Service] = {}
    for rule in http_route.instance.to_dict()["spec"].get("rules") or []:
        for backend_ref in rule.get("backendRefs") or []:
            if (backend_ref.get("kind") or "Service") != "Service":
                continue
            namespace: str = str(backend_ref.get("namespace") or http_route.namespace)
            name: str = str(backend_ref["name"])
            resolved.setdefault(
                (namespace, name),
                Service(client=client, name=name, namespace=namespace),
            )
    return list(resolved.values())


def pods_for_service(client: DynamicClient, service: Service) -> list[Pod]:
    """Return the pods selected by a Service.

    Args:
        client: Client with read access to the Service namespace.
        service: Service whose `spec.selector` is used.

    Returns:
        The matching pods; empty when the Service has no selector.
    """
    selector = service.instance.to_dict()["spec"].get("selector") or {}
    if not selector:
        return []
    label_selector = ",".join(f"{key}={value}" for key, value in sorted(selector.items()))
    return list(Pod.get(client=client, namespace=service.namespace, label_selector=label_selector))


def serving_pods_for_path(client: DynamicClient, http_route: HTTPRoute) -> list[Pod]:
    """Return the pods backing an HTTPRoute, through its existing backend Services.

    Args:
        client: Client with cluster-wide read access.
        http_route: Route whose serving workload is resolved.

    Returns:
        The pods selected by the route's backend Services.
    """
    return [
        pod
        for service in backend_services(client=client, http_route=http_route)
        if service.exists
        for pod in pods_for_service(client=client, service=service)
    ]


def pod_logs(pod: Pod) -> str:
    """Return the concatenated logs of every container in a pod.

    Args:
        pod: Pod to read logs from.

    Returns:
        All container logs joined by newlines.
    """
    return "\n".join(
        pod.log(container=container["name"]) for container in pod.instance.to_dict()["spec"].get("containers") or []
    )


def route_url(route: Route, path: str) -> str:
    """Build the external URL of a path exposed by an OpenShift Route.

    Args:
        route: Route providing the host and TLS configuration.
        path: Absolute request path.

    Returns:
        The externally reachable URL.
    """
    route_spec = route.instance.to_dict()["spec"]
    scheme = "https" if route_spec.get("tls") else "http"
    return f"{scheme}://{route_spec['host']}{path}"


def gateway_base_url(http_route: HTTPRoute) -> str:
    """Return the external base URL of the Gateway exposing an HTTPRoute.

    Args:
        http_route: Route whose `spec.hostnames` provides the external hostname.

    Returns:
        The `https://<hostname>` base URL, without a trailing slash.

    Raises:
        ValueError: If the route exposes no hostname, so it is not reachable
            from outside the cluster.
    """
    hostnames = http_route.instance.to_dict()["spec"].get("hostnames") or []
    if not hostnames:
        raise ValueError(
            f"HTTPRoute {http_route.namespace}/{http_route.name} exposes no hostname, "
            "so it is not reachable from outside the cluster"
        )
    return f"https://{hostnames[0]}"


def praxis_http_route(client: DynamicClient, path: str) -> HTTPRoute:
    """Return the single HTTPRoute publishing an API path at the public boundary.

    The owning route is resolved from the cluster so nothing about Praxis naming is
    hardcoded. A path no route declares means Praxis does not front that API here.

    Args:
        client: Client with cluster-wide read access.
        path: Absolute request path, for example `/v1/files`.

    Returns:
        The route declaring an exact match on `path`.

    Raises:
        UnexpectedResourceCountError: If more than one HTTPRoute declares `path`, making
            the public boundary ambiguous.
    """
    http_routes = http_routes_matching_path(client=client, path=path)
    if not http_routes:
        pytest.skip(
            f"No HTTPRoute declares {path}; Praxis is not the public entrypoint on this cluster. "
            f"Deploy Praxis with a Gateway API route for {path} to run this test."
        )
    if len(http_routes) > 1:
        raise UnexpectedResourceCountError(
            f"Expected exactly one HTTPRoute to own {path}, found "
            f"{[f'{route.namespace}/{route.name}' for route in http_routes]}; the public boundary is ambiguous"
        )
    return http_routes[0]


def praxis_api_url(client: DynamicClient, path: str) -> str:
    """Return the public URL of an OpenAI-compatible API path served by Praxis.

    Args:
        client: Client with cluster-wide read access.
        path: Absolute request path, for example `/v1/files`.

    Returns:
        The externally reachable URL of `path`.
    """
    return f"{gateway_base_url(http_route=praxis_http_route(client=client, path=path))}{path}"


def praxis_gateway_base_url(client: DynamicClient, paths: Iterable[str]) -> str:
    """Return the single external base URL serving every given API path.

    Each path is resolved through the HTTPRoute that owns it, so the base URL is the
    boundary a client outside the cluster actually addresses. The paths are required to
    agree: a split external boundary is a finding in itself, not something to pick a
    winner from. A path no route declares skips the test, as `praxis_http_route` does.

    Args:
        client: Client with cluster-wide read access.
        paths: Absolute request paths that must share one public hostname.

    Returns:
        The `https://<hostname>` base URL shared by all `paths`.

    Raises:
        UnexpectedResourceCountError: If the paths resolve to more than one base URL.
    """
    base_urls = {path: gateway_base_url(http_route=praxis_http_route(client=client, path=path)) for path in paths}
    if len(set(base_urls.values())) > 1:
        raise UnexpectedResourceCountError(
            f"Expected one external base URL for all API paths, found {base_urls}; "
            "the public boundary is split across hostnames"
        )
    return next(iter(base_urls.values()))
