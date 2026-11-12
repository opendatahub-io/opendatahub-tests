"""In-cluster deployments of the mock upstream servers.

The mocks run as ConfigMap-script + Deployment + Service in the test namespace
(the same layout as the validated cluster manifests). ``deploy_mock_maas_server``
additionally creates a Route so tests can query the mock's debug endpoints from
outside the cluster. The debug endpoints are guarded by a per-deployment random
token (created as a Secret and passed to the container via env), because the
recordings they return include the request headers the mock received.
"""

from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from secrets import token_urlsafe
from typing import Any

import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.resource import ResourceEditor
from ocp_resources.route import Route
from ocp_resources.secret import Secret
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

# Key of the debug-token Secret and the env variable the mock reads it from.
_MOCK_DEBUG_TOKEN_SECRET_KEY = "token"
_MOCK_DEBUG_TOKEN_ENV = "MOCK_DEBUG_TOKEN"


@dataclass
class MockUpstream:
    """Handle to a deployed mock upstream."""

    # In-cluster URL (for OGX provider configs).
    service_url: str
    # External route host (https, edge-terminated) or None when not routed.
    route_host: str | None = None
    # Token required by the mock's debug endpoints (X-Mock-Debug-Token header)
    # when the deployment created one, else None.
    debug_token: str | None = None

    @property
    def route_url(self) -> str:
        """External https URL of the mock's route (requires ``routed=True``)."""
        if self.route_host is None:
            raise ValueError("Mock upstream was deployed without a route")
        return f"https://{self.route_host}"

    def debug_headers(self) -> dict[str, str]:
        """Headers required by the mock's debug endpoints."""
        if self.debug_token is None:
            return {}
        return {"X-Mock-Debug-Token": self.debug_token}


def _mock_deployment_template(
    app_label: str,
    port: int,
    script_configmap_name: str,
    script_filename: str,
    debug_token_secret_name: str | None = None,
) -> dict:
    """Pod template for a mock server Deployment (script mounted from ConfigMap)."""
    container: dict[str, Any] = {
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
    if debug_token_secret_name:
        container["env"] = [
            {
                "name": _MOCK_DEBUG_TOKEN_ENV,
                "valueFrom": {
                    "secretKeyRef": {
                        "name": debug_token_secret_name,
                        "key": _MOCK_DEBUG_TOKEN_SECRET_KEY,
                    }
                },
            }
        ]
    return {
        "metadata": {"labels": {"app": app_label}},
        "spec": {
            "containers": [container],
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
    debug_token: str | None = None,
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
        debug_token: When set, the mock requires this token on its debug
            endpoints (deployed as a Secret, referenced via env).
    """
    script_path = _SCRIPTS_DIR / script_filename

    debug_token_secret_name = f"{name}-debug-token" if debug_token else None
    script_path = _SCRIPTS_DIR / script_filename

    debug_token_secret_name = f"{name}-debug-token" if debug_token else None
    with ExitStack() as stack:
        stack.enter_context(
            ConfigMap(
                client=client,
                namespace=namespace.name,
                name=script_configmap_name,
                data={script_filename: script_path.read_text()},
                wait_for_resource=True,
                teardown=teardown,
            )
        )
        if debug_token_secret_name:
            stack.enter_context(
                Secret(
                    client=client,
                    namespace=namespace.name,
                    name=debug_token_secret_name,
                    string_data={_MOCK_DEBUG_TOKEN_SECRET_KEY: debug_token},
                    teardown=teardown,
                )
            )
        deployment = stack.enter_context(
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
                    debug_token_secret_name=debug_token_secret_name,
                ),
                teardown=teardown,
            )
        )
        deployment.wait_for_replicas(deployed=True, timeout=300)
        service = stack.enter_context(
            Service(
                client=client,
                namespace=namespace.name,
                name=name,
                selector={"app": name},
                ports=[{"port": port, "targetPort": port, "protocol": "TCP"}],
                wait_for_resource=True,
                teardown=teardown,
            )
        )
        LOGGER.info("Deployed mock upstream", name=name, namespace=namespace.name)
        if not routed:
            yield MockUpstream(service_url=service_url)
            return

        route = stack.enter_context(
            Route(
                client=client,
                namespace=namespace.name,
                name=f"{name}-route",
                service=service.name,
                wait_for_resource=True,
                teardown=teardown,
            )
        )
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
        yield MockUpstream(service_url=service_url, route_host=route.host, debug_token=debug_token)


@contextmanager
def deploy_mock_maas_server(
    client: DynamicClient,
    namespace: Namespace,
    teardown: bool = True,
    routed: bool = True,
) -> Generator[MockUpstream, Any, Any]:
    """Deploy the mock MaaS (genai-proxy) upstream.

    The mock's debug endpoints (which expose recorded request headers, i.e.
    forwarded credentials) are guarded by a per-deployment random token.
    """
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
        debug_token=token_urlsafe(32),
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
