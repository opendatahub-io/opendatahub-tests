"""Constants shared by the OGX -> Praxis migration tests."""

# Paths of the OpenAI-compatible APIs, as declared by the Gateway API routes
# that expose them after the migration to Praxis.
RESPONSES_API_PATH: str = "/v1/responses"
EMBEDDINGS_API_PATH: str = "/v1/embeddings"
MODELS_API_PATH: str = "/v1/models"
FILES_API_PATH: str = "/v1/files"
VECTOR_STORES_API_PATH: str = "/v1/vector_stores"

# Timeouts for requests sent through the external Gateway hostname.
REQUEST_TIMEOUT_SECONDS: int = 120
PROBE_TIMEOUT_SECONDS: int = 30
