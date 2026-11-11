"""In-cluster deployments of the mock upstream servers.

The mocks run as ConfigMap-script + Deployment + Service in the test namespace
(the same layout as the validated cluster manifests). ``deploy_mock_maas_server``
additionally creates a Route so tests can query the mock's debug endpoints from
outside the cluster (the mock carries no NetworkPolicy of its own).
"""

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.resource import ResourceEditor
from ocp_resources.route import Route
from ocp_resources.service import Service

from tests.ogx.mockupstream.constants import (
    MOCK_MAAS_NAME,
    MOCK_MAAS_PORT,
    MOCK_MAAS_SCRIPT_CONFIGMAP_NAME,
    MOCK_MAAS_SERVICE_URL,
    MOCK_MCP_NAME,
    MOCK_MCP_PORT,
    MOCK_MCP_SCRIPT_CONFIGMAP_NAME,
    MOCK_MCP_SERVICE_URL,
    MOCK_SERVER_IMAGE,
)

LOGGER = structlog.get_logger(name=__name__)

_SCRIPTS_DIR = Path(__file__).parent / "scripts"


@dataclass
class MockUpstream:
    """Handle to a deployed mock upstream."""

    # In-cluster URL (for OGX provider configs).
    service_url: str
    # External route host (https, edge-terminated) or None when not routed.
    route_host: str | None = None

    @property
    def route_url(self) -> str:
        """External https URL of the mock's route (requires ``routed=True``)."""
        if self.route_host is None:
            raise ValueError("Mock upstream was deployed without a route")
        return f"https://{self.route_host}"


def _mock_deployment_template(app_label: str, port: int, script_configmap_name: str, script_filename: str) -> dict:
    """Pod template for a mock server Deployment (script mounted from ConfigMap)."""
    return {
        "metadata": {"labels": {"app": app_label}},
        "spec": {
            "containers": [
                {
                    "name": "mock-server",
                    "image": MOCK_SERVER_IMAGE,
                    "command": ["python3", f"/opt/app/{script_filename}"],
                    "ports": [{"containerPort": port, "protocol": "TCP"}],
                    "readinessProbe": {
                        "tcpSocket": {"port": port},
                        "initialDelaySeconds": 3,
                        "periodSeconds": 5,
                    },
                    "resources": {
                        "requests": {"cpu": "50m", "memory": "64Mi"},
                        "limits": {"cpu": "200m", "memory": "128Mi"},
                    },
                    "volumeMounts": [{"name": "script", "mountPath": "/opt/app", "readOnly": True}],
                }
            ],
            "volumes": [{"name": "script", "configMap": {"name": script_configmap_name}}],
        },
    }


@contextmanager
def deploy_mock_upstream(
    client: DynamicClient,
    namespace: Namespace,
    name: str,
    port: int,
    script_filename: str,
    script_configmap_name: str,
    service_url: str,
    teardown: bool = True,
    routed: bool = True,
) -> Generator[MockUpstream, Any, Any]:
    """Deploy a ConfigMap-script mock server (ConfigMap + Deployment + Service).

    Args:
        client: Kubernetes dynamic client.
        namespace: Namespace to deploy into.
        name: Name for the Deployment/Service.
        port: Container/service port.
        script_filename: Filename of the server script under ``scripts/``;
            also used as the ConfigMap key (files mount under their key) and
            in the container command, so the three cannot diverge.
        script_configmap_name: ConfigMap name carrying the server script.
        service_url: In-cluster URL returned to the caller.
        teardown: Whether to delete the resources on exit.
        routed: Whether to create an edge-terminated Route to the mock, so
            tests can reach its debug endpoints from outside the cluster.
    """
    script_path = _SCRIPTS_DIR / script_filename

    with (
        ConfigMap(
            client=client,
            namespace=namespace.name,
            name=script_configmap_name,
            data={script_filename: script_path.read_text()},
            wait_for_resource=True,
            teardown=teardown,
        ),
        Deployment(
            client=client,
            namespace=namespace.name,
            name=name,
            selector={"matchLabels": {"app": name}},
            template=_mock_deployment_template(
                app_label=name,
                port=port,
                script_configmap_name=script_configmap_name,
                script_filename=script_filename,
            ),
            teardown=teardown,
        ) as deployment,
    ):
        deployment.wait_for_replicas(deployed=True, timeout=300)
        with Service(
            client=client,
            namespace=namespace.name,
            name=name,
            selector={"app": name},
            ports=[{"port": port, "targetPort": port, "protocol": "TCP"}],
            wait_for_resource=True,
            teardown=teardown,
        ) as service:
            LOGGER.info("Deployed mock upstream", name=name, namespace=namespace.name)
            if not routed:
                yield MockUpstream(service_url=service_url)
                return

            with Route(
                client=client,
                namespace=namespace.name,
                name=f"{name}-route",
                service=service.name,
                wait_for_resource=True,
                teardown=teardown,
            ) as route:
                ResourceEditor(
                    patches={
                        route: {
                            "spec": {
                                "tls": {
                                    "termination": "edge",
                                    "insecureEdgeTerminationPolicy": "Redirect",
                                },
                                "port": {"targetPort": port},
                            },
                        }
                    }
                ).update()
                route.wait(timeout=60)
                yield MockUpstream(service_url=service_url, route_host=route.host)


@contextmanager
def deploy_mock_maas_server(
    client: DynamicClient,
    namespace: Namespace,
    teardown: bool = True,
    routed: bool = True,
) -> Generator[MockUpstream, Any, Any]:
    """Deploy the mock MaaS (genai-proxy) upstream."""
    with deploy_mock_upstream(
        client=client,
        namespace=namespace,
        name=MOCK_MAAS_NAME,
        port=MOCK_MAAS_PORT,
        script_filename="mock_maas_server.py",
        script_configmap_name=MOCK_MAAS_SCRIPT_CONFIGMAP_NAME,
        service_url=MOCK_MAAS_SERVICE_URL,
        teardown=teardown,
        routed=routed,
    ) as upstream:
        yield upstream


@contextmanager
def deploy_mock_mcp_server(
    client: DynamicClient,
    namespace: Namespace,
    teardown: bool = True,
    routed: bool = False,
) -> Generator[MockUpstream, Any, Any]:
    """Deploy the mock MCP server (no route by default: OGX reaches it in-cluster)."""
    with deploy_mock_upstream(
        client=client,
        namespace=namespace,
        name=MOCK_MCP_NAME,
        port=MOCK_MCP_PORT,
        script_filename="mock_mcp_server.py",
        script_configmap_name=MOCK_MCP_SCRIPT_CONFIGMAP_NAME,
        service_url=MOCK_MCP_SERVICE_URL,
        teardown=teardown,
        routed=routed,
    ) as upstream:
        yield upstream


__all__ = [
    "MockUpstream",
    "deploy_mock_maas_server",
    "deploy_mock_mcp_server",
]
