"""In-cluster discovery helpers for observability operands and services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kubernetes.dynamic import DynamicClient


@dataclass(frozen=True)
class ServiceEndpoint:
    """A discovered service endpoint without credentials or route bodies."""

    namespace: str
    name: str
    port: int
    scheme: str = "https"

    @property
    def base_url(self) -> str:
        """Return the in-cluster DNS URL used by signal query adapters."""
        return f"{self.scheme}://{self.name}.{self.namespace}.svc:{self.port}"

    def to_dict(self) -> dict[str, object]:
        """Return sanitized discovery evidence."""
        return {"namespace": self.namespace, "name": self.name, "port": self.port, "scheme": self.scheme}


def discover_operand_namespace(
    admin_client: DynamicClient,
    *,
    api_version: str,
    kind: str,
    label_selector: str | None = None,
) -> str:
    """Discover the sole namespace containing an active operand resource.

    The API version, kind, and selector are supplied by the active release contract;
    no monitoring namespace is embedded in this helper.
    """
    resource_getter = admin_client.resources.get
    resource_api = resource_getter(api_version=api_version, kind=kind)
    kwargs = {"label_selector": label_selector} if label_selector else {}
    response = resource_api.get(**kwargs)
    resources = getattr(response, "items", None)
    if not isinstance(resources, list):
        raise TypeError("operand list response must expose a list through its items attribute")
    namespaces = {
        str(metadata.namespace)
        for resource in resources
        if (metadata := getattr(resource, "metadata", None)) is not None and getattr(metadata, "namespace", None)
    }
    if len(namespaces) != 1:
        raise LookupError(f"expected one {kind} operand namespace, found {sorted(namespaces)}")
    return next(iter(namespaces))


def discover_service_endpoint(
    admin_client: DynamicClient,
    *,
    namespace: str,
    name: str,
    port_name: str | None = None,
    scheme: str = "https",
) -> ServiceEndpoint:
    """Discover a service port and construct its in-cluster endpoint."""
    from ocp_resources.service import Service

    service = Service(client=admin_client, namespace=namespace, name=name)
    if not service.exists:
        raise LookupError(f"service {namespace}/{name} does not exist")
    ports = getattr(getattr(service.instance, "spec", None), "ports", None) or []
    matching_ports = [port for port in ports if port_name is None or getattr(port, "name", None) == port_name]
    if len(matching_ports) != 1:
        raise LookupError(f"expected one matching port for service {namespace}/{name}, found {len(matching_ports)}")
    port = getattr(matching_ports[0], "port", None)
    if not isinstance(port, int) or port <= 0:
        raise LookupError(f"service {namespace}/{name} has no usable port")
    return ServiceEndpoint(namespace=namespace, name=name, port=port, scheme=scheme)
