"""Constants for the in-cluster mock upstream deployments."""

import os

from tests.ogx.image_constants import OgxImages

# Container image for both mock servers (digest-pinned in image_constants.py).
# The OGX_MOCK_SERVER_IMAGE env var allows pointing at a mirrored location
# (e.g. a disconnected registry); the override must itself be digest-pinned —
# mutable tags are rejected per the image policy.
MOCK_SERVER_IMAGE = os.getenv("OGX_MOCK_SERVER_IMAGE", OgxImages.MOCK_SERVER)
if "@sha256:" not in MOCK_SERVER_IMAGE:
    raise ValueError(
        f"OGX_MOCK_SERVER_IMAGE must reference an image digest (registry/repo@sha256:...), "
        f"mutable tags are not allowed, got: {MOCK_SERVER_IMAGE!r}"
    )

# Non-empty api_key for the passthrough provider in suites where the
# per-request credential flow is NOT under test (MCP tool runtime, OTel):
# OGX's internal OpenAI-compatible client requires one, while the MaaS
# passthrough suite deploys with an empty key on purpose.
MOCK_INFERENCE_API_KEY = "mock-inference-key"  # pragma: allowlist secret

# --- mock MaaS / genai-proxy upstream -------------------------------------
MOCK_MAAS_NAME = "mock-maas-server"
MOCK_MAAS_SCRIPT_CONFIGMAP_NAME = "mock-maas-script"
MOCK_MAAS_PORT = 9999
MOCK_MAAS_ROUTE_NAME = "mock-maas"
# In-cluster URL for the passthrough provider's base_url (OGX appends /v1).
MOCK_MAAS_SERVICE_URL = f"http://{MOCK_MAAS_NAME}:{MOCK_MAAS_PORT}"
# Model served by the mock (registered in the OGX run config). OGX reports
# registered models with the provider prefix; API calls must use the
# qualified id.
MOCK_MAAS_MODEL_ID = "test-maas-model"

# --- genai-bff-proxy remote::passthrough provider ---------------------------
# The inference path the Gen AI BFF registers for Playgrounds (names must
# match the odh-dashboard BFF).
PASSTHROUGH_PROVIDER_ID = "genai-bff-proxy"
PASSTHROUGH_PROVIDER_TYPE = "remote::passthrough"
# Model id as OGX reports it (registered models carry the provider prefix);
# API calls must use the qualified form.
MOCK_MAAS_QUALIFIED_MODEL_ID = f"{PASSTHROUGH_PROVIDER_ID}/{MOCK_MAAS_MODEL_ID}"
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
