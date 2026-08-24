MLFLOW_BFF_PATH: str = "/_bff/mlflow/api/v1/mcp-registry/servers"
MLFLOW_BFF_REGISTER_PATH: str = "/_bff/mlflow/api/v1/mcp-registry/register"
MODEL_REGISTRY_BFF_PATH: str = "/model-registry/api/v1/mcp_deployments"
REGISTRY_SERVER_ANNOTATION: str = "mcp.opendatahub.io/registry-server"
REGISTRY_VERSION_ANNOTATION: str = "mcp.opendatahub.io/registry-version"
DEPLOYABLE_MCP_REGISTRY_NAMESPACE: str = "odh-qe"
DEPLOYABLE_MCP_VERSION: str = "0.0.65"
DEPLOYABLE_MCP_PORT: int = 8080
DEPLOYABLE_MCP_PATH: str = "/mcp"
DEPLOYABLE_MCP_SOURCE_NAME_PREFIX: str = "Test MCP Registry"
DEPLOYABLE_MCP_SERVER_TITLE: str = "Kubernetes MCP Server"
DEPLOYABLE_MCP_SERVER_DESCRIPTION: str = (
    "Deployable Kubernetes MCP server used for AI Hub registry integration testing."
)
DEPLOYABLE_MCP_PROVIDER: str = "OpenDataHub QE"
DEPLOYABLE_MCP_LICENSE: str = "Apache-2.0"
DEPLOYABLE_MCP_TAGS: tuple[str, ...] = ("kubernetes", "registry", "integration")
DEPLOYABLE_MCP_TIMESTAMP_MS: str = "1755523200000"
DEPLOYABLE_MCP_EXPECTED_TOOL_NAMES: tuple[str, ...] = (
    "resources_list",
    "pods_list",
    "namespaces_list",
)
DASHBOARD_LOCAL_PORT: int = 18446
DASHBOARD_REMOTE_PORT: int = 8443
