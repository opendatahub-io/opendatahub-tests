"""Constants shared by the OGX -> Praxis migration tests."""

from typing import Any

# Paths of the OpenAI-compatible APIs, as declared by the Gateway API routes
# that expose them after the migration to Praxis.
RESPONSES_API_PATH: str = "/v1/responses"
EMBEDDINGS_API_PATH: str = "/v1/embeddings"
MODELS_API_PATH: str = "/v1/models"
FILES_API_PATH: str = "/v1/files"
VECTOR_STORES_API_PATH: str = "/v1/vector_stores"
CONVERSATIONS_API_PATH: str = "/v1/conversations"

# Service port OGX listens on for traffic delegated by Praxis. In Praxis-fronted mode the
# operator-managed NetworkPolicy admits this port only from Praxis and the operator namespace.
OGX_DELEGATION_PORT: int = 8321

# Pod labels identifying Praxis, used as the OGXServer praxisSelector.
PRAXIS_POD_LABELS: dict[str, str] = {"app": "payload-processing"}

# Timeout for requests sent through the external Gateway hostname.
REQUEST_TIMEOUT_SECONDS: int = 30
PROBE_TIMEOUT_SECONDS: int = 30

# Parameters for the namespace the tests deploy into.
NAMESPACE_PARAMS: dict[str, Any] = {"name": "test-ogx-to-praxis"}

# Parameters for the OGXServer the tests deploy.
OGX_SERVER_PARAMS: dict[str, Any] = {
    "ogx_storage_size": "2Gi",
    "vector_io_provider": "pgvector",
    "files_provider": "local",
}

# Parameters for an OGXServer in Praxis-fronted internal-only mode. External access is
# requested on purpose: in Praxis mode the operator must not honor it.
PRAXIS_MODE_OGX_SERVER_PARAMS: dict[str, Any] = {
    **OGX_SERVER_PARAMS,
    "network": {"externalAccess": {"enabled": True}},
    "praxis_mode": {
        "enabled": True,
        "praxisSelector": {"podSelector": {"matchLabels": PRAXIS_POD_LABELS}},
    },
}
