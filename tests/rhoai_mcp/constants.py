RHOAI_MCP_APP_NAME: str = "rhoai-mcp"
RHOAI_MCP_NAMESPACE: str = "test-rhoai-mcp"
RHOAI_MCP_PORT: int = 8000
RHOAI_MCP_HEALTH_PATH: str = "/health"
RHOAI_MCP_ENDPOINT_PATH: str = "/mcp"
RHOAI_MCP_CLUSTERROLE_NAME: str = "test-rhoai-mcp"

RHOAI_MCP_EXPECTED_SERVING_TOOLS: tuple[str, ...] = (
    "list_inference_services",
    "get_inference_service",
    "deploy_model",
    "delete_inference_service",
    "list_serving_runtimes",
    "create_serving_runtime",
    "get_model_endpoint",
    "prepare_model_deployment",
    "check_deployment_prerequisites",
    "estimate_serving_resources",
    "recommend_serving_runtime",
    "test_model_endpoint",
)

RHOAI_MCP_EXPECTED_CATALOG_TOOLS: tuple[str, ...] = (
    "list_catalog_sources",
    "get_catalog_model_artifacts",
)

RHOAI_MCP_EXPECTED_PROMPTS: tuple[str, ...] = (
    "explore-cluster",
    "deploy-model",
    "deploy-llm",
    "find-gpus",
)

RHOAI_MCP_RBAC_READER_ROLE_NAME: str = "test-rhoai-mcp-reader"
RHOAI_MCP_RBAC_DEPLOYER_ROLE_NAME: str = "test-rhoai-mcp-deployer"

RHOAI_MCP_INFERENCE_READ_TOOLS: tuple[str, ...] = (
    "list_inference_services",
    "get_inference_service",
    "list_serving_runtimes",
    "get_model_endpoint",
    "test_model_endpoint",
    "check_deployment_prerequisites",
    "estimate_serving_resources",
    "recommend_serving_runtime",
)

RHOAI_MCP_INFERENCE_DEPLOY_TOOLS: tuple[str, ...] = (
    "deploy_model",
    "prepare_model_deployment",
)

RHOAI_MCP_INFERENCE_RESTRICTED_TOOLS: tuple[str, ...] = (
    "delete_inference_service",
    "create_serving_runtime",
)
