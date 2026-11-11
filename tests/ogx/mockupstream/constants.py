"""Constants for the in-cluster mock upstream deployments."""

import os

# Container image for both mock servers. Overridable so disconnected lanes can
# point at a mirrored location (the image is already part of the disconnected
# image set: registry.redhat.io/ubi9/python-312). A digest may be pinned.
MOCK_SERVER_IMAGE = os.getenv("OGX_MOCK_SERVER_IMAGE", "registry.access.redhat.com/ubi9/python-312:latest")

# --- mock MaaS / genai-proxy upstream -------------------------------------
MOCK_MAAS_NAME = "mock-maas-server"
MOCK_MAAS_SCRIPT_CONFIGMAP_NAME = "mock-maas-script"
MOCK_MAAS_SCRIPT_KEY = "server.py"
MOCK_MAAS_PORT = 9999
MOCK_MAAS_ROUTE_NAME = "mock-maas"
# In-cluster URL for the passthrough provider's base_url (OGX appends /v1).
MOCK_MAAS_SERVICE_URL = f"http://{MOCK_MAAS_NAME}:{MOCK_MAAS_PORT}"
# Model served by the mock (registered in the OGX run config).
MOCK_MAAS_MODEL_ID = "test-maas-model"

# --- genai-bff-proxy remote::passthrough provider ---------------------------
# The inference path the Gen AI BFF registers for Playgrounds (names must
# match the odh-dashboard BFF).
PASSTHROUGH_PROVIDER_ID = "genai-bff-proxy"
PASSTHROUGH_PROVIDER_TYPE = "remote::passthrough"
# X-OGX-Provider-Data JSON keys mapped to outbound HTTP headers so per-request
# credentials (e.g. MaaS tokens) flow through OGX without OGX understanding them.
PASSTHROUGH_FORWARD_HEADERS = {
    "maas_subscription": "X-MaaS-Subscription",
    "inference_model_source_type": "X-Inference-Model-Source-Type",
    "traceparent": "traceparent",
    "tracestate": "tracestate",
    "baggage": "baggage",
}

# --- mock MCP server -------------------------------------------------------
MOCK_MCP_NAME = "mock-mcp-server"
MOCK_MCP_SCRIPT_CONFIGMAP_NAME = "mock-mcp-script"
MOCK_MCP_SCRIPT_KEY = "server.py"
MOCK_MCP_PORT = 9000
MOCK_MCP_ENDPOINT_PATH = "/mcp"
MOCK_MCP_SERVICE_URL = f"http://{MOCK_MCP_NAME}:{MOCK_MCP_PORT}{MOCK_MCP_ENDPOINT_PATH}"
# The single tool the mock MCP server exposes.
MOCK_MCP_TOOL_NAME = "get_weather"
# connector_id the OGX run config registers the mock MCP server under.
MOCK_MCP_CONNECTOR_ID = "test-mcp-server"
MOCK_MCP_SERVER_LABEL = "test-mcp"
# Fixed tool result the mock returns (deterministic assertion target).
MOCK_MCP_TOOL_RESULT_TEXT = "Weather in Boston: sunny, 21C"
