"""Constants for EvalHub MCP server integration tests.

Mirrors operator deployment conventions in trustyai-service-operator/controllers/evalhub/.
"""

EVALHUB_MCP_CR_NAME: str = "evalhub-mcp-mt"

EVALHUB_MCP_APP_LABEL: str = "eval-hub"
EVALHUB_MCP_COMPONENT_LABEL: str = "mcp"
EVALHUB_MCP_CONTAINER_NAME: str = "evalhub-mcp"
EVALHUB_MCP_KUBE_RBAC_PROXY_CONTAINER: str = "kube-rbac-proxy"

EVALHUB_MCP_SERVICE_PORT: int = 8443
EVALHUB_MCP_HEALTH_PATH: str = "/health"
EVALHUB_MCP_HEALTH_STATUS_OK: str = "ok"

EVALHUB_MCP_PROTOCOL_VERSION: str = "2024-11-05"
EVALHUB_MCP_CLIENT_NAME: str = "opendatahub-tests"
EVALHUB_MCP_CLIENT_VERSION: str = "1.0"

# MCP server metadata (internal/evalhub_mcp/server/server.go)
EVALHUB_MCP_SERVER_NAME: str = "evalhub-mcp"

# Tools (internal/evalhub_mcp/server/tools.go)
EVALHUB_MCP_TOOLS: tuple[str, ...] = (
    "submit_evaluation",
    "cancel_job",
    "get_job_status",
    "discover_providers",
)

# Static resources (internal/evalhub_mcp/server/resources.go, version_resource.go)
EVALHUB_MCP_RESOURCE_URIS: tuple[str, ...] = (
    "evalhub://providers",
    "evalhub://benchmarks",
    "evalhub://collections",
    "evalhub://jobs",
    "evalhub://server/version",
)

# Prompts (internal/evalhub_mcp/server/prompts.go)
EVALHUB_MCP_PROMPTS: tuple[str, ...] = (
    "edd_workflow",
    "evaluate_model",
    "compare_runs",
)

# RBAC for external MCP clients (trustyai-service-operator/controllers/evalhub/mcp_configmap.go)
EVALHUB_MCP_PROXY_RESOURCE: str = "evalhubs/proxy"

# Operator resource naming ({cr-name}-mcp pattern)
EVALHUB_MCP_DEPLOYMENT_SUFFIX: str = "-mcp"
EVALHUB_MCP_CONFIGMAP_SUFFIX: str = "-mcp-config"

# Known provider/benchmark IDs from eval-hub defaults
EVALHUB_MCP_DEFAULT_PROVIDER_ID: str = "lm_evaluation_harness"
EVALHUB_MCP_DEFAULT_BENCHMARK_ID: str = "arc_easy"
EVALHUB_MCP_DEFAULT_COLLECTION_ID: str = "leaderboard-v2"

# EDD workflow application types (internal/evalhub_mcp/server/prompts.go)
EVALHUB_MCP_EDD_APPLICATION_TYPES: tuple[str, ...] = (
    "rag",
    "agent",
    "safety",
    "classifier",
)

# Resource URI templates for dynamic reads
EVALHUB_MCP_PROVIDER_URI_TEMPLATE: str = "evalhub://providers/{provider_id}"
EVALHUB_MCP_BENCHMARK_URI_TEMPLATE: str = "evalhub://benchmarks/{benchmark_id}"
EVALHUB_MCP_COLLECTION_URI_TEMPLATE: str = "evalhub://collections/{collection_id}"
EVALHUB_MCP_JOB_URI_TEMPLATE: str = "evalhub://jobs/{job_id}"
EVALHUB_MCP_JOBS_BY_STATUS_URI_TEMPLATE: str = "evalhub://jobs?status={status}"
