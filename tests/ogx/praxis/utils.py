"""Helpers shared by the OGX -> Praxis migration tests.

The Gateway helpers resolve what actually serves an API path from the cluster
itself: a path is matched against the HTTPRoutes that declare it, and from a
matching route the backend Services and their pods are resolved. Nothing about
Praxis naming is hardcoded, so a route pointing at the wrong backend cannot pass
unnoticed.
"""

import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.pod import Pod
from ocp_resources.route import Route
from ocp_resources.service import Service

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
