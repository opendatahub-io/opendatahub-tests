from collections.abc import Generator
from typing import Any

import pytest
from fastmcp.client.transports import StreamableHttpTransport
from kubernetes.dynamic import DynamicClient
from ocp_resources.cluster_role import ClusterRole
from ocp_resources.cluster_role_binding import ClusterRoleBinding
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route
from ocp_resources.service import Service
from ocp_resources.service_account import ServiceAccount

from tests.rhoai_mcp.constants import (
    RHOAI_MCP_APP_NAME,
    RHOAI_MCP_CLUSTERROLE_NAME,
    RHOAI_MCP_ENDPOINT_PATH,
    RHOAI_MCP_HEALTH_PATH,
    RHOAI_MCP_NAMESPACE,
    RHOAI_MCP_PORT,
    RHOAI_MCP_RBAC_DEPLOYER_ROLE_NAME,
    RHOAI_MCP_RBAC_READER_ROLE_NAME,
)
from tests.rhoai_mcp.utils import (
    DEPLOYMENT_TEMPLATE,
    probe_health,
)
from utilities.certificates_utils import create_ca_bundle_file
from utilities.infra import create_inference_token, create_ns


@pytest.fixture(scope="class")
def rhoai_mcp_namespace(
    admin_client: DynamicClient,
    teardown_resources: bool,
) -> Generator[Namespace, Any, Any]:
    """Namespace for rhoai-mcp deployment."""
    with create_ns(
        admin_client=admin_client,
        name=RHOAI_MCP_NAMESPACE,
        teardown=teardown_resources,
    ) as ns:
        yield ns


@pytest.fixture(scope="class")
def rhoai_mcp_service_account(
    admin_client: DynamicClient,
    rhoai_mcp_namespace: Namespace,
) -> Generator[ServiceAccount, Any, Any]:
    """ServiceAccount for the rhoai-mcp server."""
    with ServiceAccount(
        client=admin_client,
        name=RHOAI_MCP_APP_NAME,
        namespace=rhoai_mcp_namespace.name,
        label={
            "app.kubernetes.io/component": "serviceaccount",
            "app.kubernetes.io/name": RHOAI_MCP_APP_NAME,
        },
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def rhoai_mcp_cluster_role(
    admin_client: DynamicClient,
    rhoai_mcp_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[ClusterRole, Any, Any]:
    """ClusterRole granting impersonation and token review permissions."""
    with ClusterRole(
        client=admin_client,
        name=RHOAI_MCP_CLUSTERROLE_NAME,
        teardown=teardown_resources,
        rules=[
            {
                "apiGroups": [""],
                "resources": ["users", "groups", "serviceaccounts"],
                "verbs": ["impersonate"],
            },
            {
                "apiGroups": ["authentication.k8s.io"],
                "resources": ["tokenreviews"],
                "verbs": ["create"],
            },
            {
                "apiGroups": ["authorization.k8s.io"],
                "resources": ["subjectaccessreviews"],
                "verbs": ["create"],
            },
            {
                "apiGroups": ["user.openshift.io"],
                "resources": ["users"],
                "verbs": ["get"],
            },
        ],
    ) as cr:
        yield cr


@pytest.fixture(scope="class")
def rhoai_mcp_cluster_role_binding(
    admin_client: DynamicClient,
    rhoai_mcp_namespace: Namespace,
    rhoai_mcp_service_account: ServiceAccount,
    rhoai_mcp_cluster_role: ClusterRole,
    teardown_resources: bool,
) -> Generator[ClusterRoleBinding, Any, Any]:
    """ClusterRoleBinding binding rhoai-mcp SA to its ClusterRole."""
    with ClusterRoleBinding(
        client=admin_client,
        name=RHOAI_MCP_CLUSTERROLE_NAME,
        teardown=teardown_resources,
        cluster_role=rhoai_mcp_cluster_role.name,
        subjects=[
            {
                "kind": "ServiceAccount",
                "name": rhoai_mcp_service_account.name,
                "namespace": rhoai_mcp_namespace.name,
            }
        ],
    ) as crb:
        yield crb


@pytest.fixture(scope="class")
def rhoai_mcp_config(
    admin_client: DynamicClient,
    rhoai_mcp_namespace: Namespace,
) -> Generator[ConfigMap, Any, Any]:
    """ConfigMap with rhoai-mcp server configuration."""
    with ConfigMap(
        client=admin_client,
        name=f"{RHOAI_MCP_APP_NAME}-config",
        namespace=rhoai_mcp_namespace.name,
        data={
            "RHOAI_MCP_HOST": "0.0.0.0",
            "RHOAI_MCP_PORT": str(RHOAI_MCP_PORT),
            "RHOAI_MCP_LOG_LEVEL": "INFO",
            "RHOAI_MCP_TRANSPORT": "streamable-http",
            "RHOAI_MCP_AUTH_MODE": "auto",
            "RHOAI_MCP_OIDC_ENABLED": "true",
            "RHOAI_MCP_OIDC_TOKEN_MODE": "token-review",
            "RHOAI_MCP_READ_ONLY_MODE": "false",
            "RHOAI_MCP_ENABLE_DANGEROUS_OPERATIONS": "false",
        },
    ) as cm:
        yield cm


@pytest.fixture(scope="class")
def rhoai_mcp_service(
    admin_client: DynamicClient,
    rhoai_mcp_namespace: Namespace,
    rhoai_mcp_config: ConfigMap,
) -> Generator[Service, Any, Any]:
    """Service fronting the rhoai-mcp Deployment."""
    with Service(
        client=admin_client,
        name=RHOAI_MCP_APP_NAME,
        namespace=rhoai_mcp_namespace.name,
        ports=[
            {
                "name": "http",
                "port": RHOAI_MCP_PORT,
                "protocol": "TCP",
                "targetPort": "http",
            }
        ],
        selector={
            "app.kubernetes.io/component": "server",
            "app.kubernetes.io/name": RHOAI_MCP_APP_NAME,
        },
    ) as svc:
        yield svc


@pytest.fixture(scope="class")
def rhoai_mcp_deployment(
    admin_client: DynamicClient,
    rhoai_mcp_namespace: Namespace,
    rhoai_mcp_service_account: ServiceAccount,
    rhoai_mcp_cluster_role_binding: ClusterRoleBinding,
    rhoai_mcp_config: ConfigMap,
    rhoai_mcp_service: Service,
) -> Generator[Deployment, Any, Any]:
    """Deployment for the rhoai-mcp server."""
    labels = {
        "app.kubernetes.io/component": "server",
        "app.kubernetes.io/name": RHOAI_MCP_APP_NAME,
    }
    with Deployment(
        client=admin_client,
        name=RHOAI_MCP_APP_NAME,
        namespace=rhoai_mcp_namespace.name,
        replicas=1,
        label=labels,
        selector={"matchLabels": labels},
        template=DEPLOYMENT_TEMPLATE,
    ) as deployment:
        deployment.wait_for_replicas(timeout=300)
        yield deployment


@pytest.fixture(scope="class")
def rhoai_mcp_route(
    admin_client: DynamicClient,
    rhoai_mcp_namespace: Namespace,
    rhoai_mcp_deployment: Deployment,
) -> Generator[Route, Any, Any]:
    """Route with edge TLS termination for the rhoai-mcp service."""
    with Route(
        client=admin_client,
        kind_dict={
            "apiVersion": "route.openshift.io/v1",
            "kind": "Route",
            "metadata": {
                "name": RHOAI_MCP_APP_NAME,
                "namespace": rhoai_mcp_namespace.name,
                "annotations": {
                    "haproxy.router.openshift.io/timeout": "300s",
                },
                "labels": {
                    "app.kubernetes.io/component": "route",
                    "app.kubernetes.io/name": RHOAI_MCP_APP_NAME,
                },
            },
            "spec": {
                "port": {"targetPort": "http"},
                "tls": {
                    "termination": "edge",
                    "insecureEdgeTerminationPolicy": "Redirect",
                },
                "to": {
                    "kind": "Service",
                    "name": RHOAI_MCP_APP_NAME,
                    "weight": 100,
                },
                "wildcardPolicy": "None",
            },
        },
    ) as route:
        yield route


@pytest.fixture(scope="class")
def rhoai_mcp_transport(
    rhoai_mcp_endpoint_url: str,
    rhoai_mcp_ca_bundle: str,
    rhoai_mcp_ready: None,
    current_client_token: str,
) -> StreamableHttpTransport:
    """Configured StreamableHttpTransport for FastMCP Client."""
    return StreamableHttpTransport(
        url=rhoai_mcp_endpoint_url,
        auth=str(current_client_token),
        verify=rhoai_mcp_ca_bundle,
    )


@pytest.fixture(scope="class")
def rhoai_mcp_ca_bundle(admin_client: DynamicClient) -> str:
    """CA bundle file for verifying TLS on the rhoai-mcp route."""
    return create_ca_bundle_file(client=admin_client)


@pytest.fixture(scope="class")
def rhoai_mcp_base_url(rhoai_mcp_route: Route) -> str:
    """Base URL (scheme + host) for the rhoai-mcp route."""
    return f"https://{rhoai_mcp_route.host}"


@pytest.fixture(scope="class")
def rhoai_mcp_endpoint_url(rhoai_mcp_base_url: str) -> str:
    """Full URL for the rhoai-mcp streamable-http endpoint."""
    return f"{rhoai_mcp_base_url}{RHOAI_MCP_ENDPOINT_PATH}"


@pytest.fixture(scope="class")
def rhoai_mcp_ready(
    rhoai_mcp_base_url: str,
    rhoai_mcp_ca_bundle: str,
) -> None:
    """Wait until the rhoai-mcp health endpoint responds on the route."""
    probe_health(
        url=f"{rhoai_mcp_base_url}{RHOAI_MCP_HEALTH_PATH}",
        ca_bundle_file=rhoai_mcp_ca_bundle,
    )


_KSERVE_API_GROUP = "serving.kserve.io"


# RBAC test personas – reader (get/list on ISVC/SR) and deployer (+ create on ISVC/PVC/Secrets)
@pytest.fixture(scope="class")
def rbac_reader_sa(
    admin_client: DynamicClient,
    rhoai_mcp_namespace: Namespace,
) -> Generator[ServiceAccount, Any, Any]:
    """ServiceAccount representing a read-only inference viewer."""
    with ServiceAccount(
        client=admin_client,
        name="rhoai-mcp-reader",
        namespace=rhoai_mcp_namespace.name,
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def rbac_reader_cluster_role(
    admin_client: DynamicClient,
    teardown_resources: bool,
) -> Generator[ClusterRole, Any, Any]:
    """ClusterRole granting get/list on InferenceServices and ServingRuntimes."""
    with ClusterRole(
        client=admin_client,
        name=RHOAI_MCP_RBAC_READER_ROLE_NAME,
        teardown=teardown_resources,
        rules=[
            {
                "apiGroups": [_KSERVE_API_GROUP],
                "resources": ["inferenceservices", "servingruntimes"],
                "verbs": ["get", "list"],
            },
        ],
    ) as cr:
        yield cr


@pytest.fixture(scope="class")
def rbac_reader_crb(
    admin_client: DynamicClient,
    rhoai_mcp_namespace: Namespace,
    rbac_reader_sa: ServiceAccount,
    rbac_reader_cluster_role: ClusterRole,
    teardown_resources: bool,
) -> Generator[ClusterRoleBinding, Any, Any]:
    """Bind reader SA to its ClusterRole."""
    with ClusterRoleBinding(
        client=admin_client,
        name=RHOAI_MCP_RBAC_READER_ROLE_NAME,
        teardown=teardown_resources,
        cluster_role=rbac_reader_cluster_role.name,
        subjects=[
            {
                "kind": "ServiceAccount",
                "name": rbac_reader_sa.name,
                "namespace": rhoai_mcp_namespace.name,
            }
        ],
    ) as crb:
        yield crb


@pytest.fixture(scope="class")
def rbac_reader_token(
    rbac_reader_sa: ServiceAccount,
    rbac_reader_crb: ClusterRoleBinding,
) -> str:
    """Short-lived token for the reader ServiceAccount."""
    return create_inference_token(model_service_account=rbac_reader_sa)


@pytest.fixture(scope="class")
def rbac_reader_transport(
    rhoai_mcp_endpoint_url: str,
    rhoai_mcp_ca_bundle: str,
    rhoai_mcp_ready: None,
    rbac_reader_token: str,
) -> StreamableHttpTransport:
    """MCP transport authenticated as the reader persona."""
    return StreamableHttpTransport(
        url=rhoai_mcp_endpoint_url,
        auth=rbac_reader_token,
        verify=rhoai_mcp_ca_bundle,
    )


@pytest.fixture(scope="class")
def rbac_deployer_sa(
    admin_client: DynamicClient,
    rhoai_mcp_namespace: Namespace,
) -> Generator[ServiceAccount, Any, Any]:
    """ServiceAccount representing a user who can also deploy models."""
    with ServiceAccount(
        client=admin_client,
        name="rhoai-mcp-deployer",
        namespace=rhoai_mcp_namespace.name,
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def rbac_deployer_cluster_role(
    admin_client: DynamicClient,
    teardown_resources: bool,
) -> Generator[ClusterRole, Any, Any]:
    """ClusterRole: read+create on ISVC, read on SR, read/create on PVCs and Secrets."""
    with ClusterRole(
        client=admin_client,
        name=RHOAI_MCP_RBAC_DEPLOYER_ROLE_NAME,
        teardown=teardown_resources,
        rules=[
            {
                "apiGroups": [_KSERVE_API_GROUP],
                "resources": ["inferenceservices"],
                "verbs": ["get", "list", "create"],
            },
            {
                "apiGroups": [_KSERVE_API_GROUP],
                "resources": ["servingruntimes"],
                "verbs": ["get", "list"],
            },
            {
                "apiGroups": [""],
                "resources": ["persistentvolumeclaims", "secrets"],
                "verbs": ["get", "list", "create"],
            },
        ],
    ) as cr:
        yield cr


@pytest.fixture(scope="class")
def rbac_deployer_crb(
    admin_client: DynamicClient,
    rhoai_mcp_namespace: Namespace,
    rbac_deployer_sa: ServiceAccount,
    rbac_deployer_cluster_role: ClusterRole,
    teardown_resources: bool,
) -> Generator[ClusterRoleBinding, Any, Any]:
    """Bind deployer SA to its ClusterRole."""
    with ClusterRoleBinding(
        client=admin_client,
        name=RHOAI_MCP_RBAC_DEPLOYER_ROLE_NAME,
        teardown=teardown_resources,
        cluster_role=rbac_deployer_cluster_role.name,
        subjects=[
            {
                "kind": "ServiceAccount",
                "name": rbac_deployer_sa.name,
                "namespace": rhoai_mcp_namespace.name,
            }
        ],
    ) as crb:
        yield crb


@pytest.fixture(scope="class")
def rbac_deployer_token(
    rbac_deployer_sa: ServiceAccount,
    rbac_deployer_crb: ClusterRoleBinding,
) -> str:
    """Short-lived token for the deployer ServiceAccount."""
    return create_inference_token(model_service_account=rbac_deployer_sa)


@pytest.fixture(scope="class")
def rbac_deployer_transport(
    rhoai_mcp_endpoint_url: str,
    rhoai_mcp_ca_bundle: str,
    rhoai_mcp_ready: None,
    rbac_deployer_token: str,
) -> StreamableHttpTransport:
    """MCP transport authenticated as the deployer persona."""
    return StreamableHttpTransport(
        url=rhoai_mcp_endpoint_url,
        auth=rbac_deployer_token,
        verify=rhoai_mcp_ca_bundle,
    )
