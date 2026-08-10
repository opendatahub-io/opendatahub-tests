import os
from typing import NamedTuple

import semver
from ogx_client.types import Model
from semver import VersionInfo

from utilities.image_constants import SharedImages


class ModelInfo(NamedTuple):
    """Container for model information from OGX client."""

    model_id: str
    embedding_model: Model
    embedding_dimension: int  # API returns integer (e.g., 768)


HTTPS_PROXY: str = os.getenv("SQUID_HTTPS_PROXY", "")

# OGX_CLIENT_VERIFY_SSL is false by default to be able to test with Self-Signed certificates
OGX_CLIENT_VERIFY_SSL = os.getenv("OGX_CLIENT_VERIFY_SSL", "false").lower() == "true"
OGX_CORE_POD_FILTER: str = "app=ogx"
OGX_OPENSHIFT_MINIMAL_VERSION: VersionInfo = semver.VersionInfo.parse("4.17.0")

POSTGRES_IMAGE = os.getenv("OGX_VECTOR_IO_POSTGRES_IMAGE", SharedImages.POSTGRESQL_15)
POSTGRESQL_USER = os.getenv("OGX_VECTOR_IO_POSTGRESQL_USER", "ps_user")
POSTGRESQL_PASSWORD = os.getenv("OGX_VECTOR_IO_POSTGRESQL_PASSWORD", "ps_password")

OGX_CORE_INFERENCE_MODEL = os.getenv("OGX_CORE_INFERENCE_MODEL", "")
OGX_CORE_VLLM_URL = os.getenv("OGX_CORE_VLLM_URL", "")
OGX_CORE_VLLM_API_TOKEN = os.getenv("OGX_CORE_VLLM_API_TOKEN", "")
OGX_CORE_VLLM_MAX_TOKENS = os.getenv("OGX_CORE_VLLM_MAX_TOKENS", "16384")
OGX_CORE_VLLM_TLS_VERIFY = os.getenv("OGX_CORE_VLLM_TLS_VERIFY", "true")

OGX_CORE_EMBEDDING_MODEL = os.getenv("OGX_CORE_EMBEDDING_MODEL", "nomic-embed-text-v1-5")
OGX_CORE_EMBEDDING_PROVIDER_MODEL_ID = os.getenv("OGX_CORE_EMBEDDING_PROVIDER_MODEL_ID", "nomic-embed-text-v1-5")
OGX_CORE_VLLM_EMBEDDING_URL = os.getenv(
    "OGX_CORE_VLLM_EMBEDDING_URL", "https://nomic-embed-text-v1-5.example.com:443/v1"
)
OGX_CORE_VLLM_EMBEDDING_API_TOKEN = os.getenv("OGX_CORE_VLLM_EMBEDDING_API_TOKEN", "fake")
OGX_CORE_VLLM_EMBEDDING_MAX_TOKENS = os.getenv("OGX_CORE_VLLM_EMBEDDING_MAX_TOKENS", "8192")
OGX_CORE_VLLM_EMBEDDING_TLS_VERIFY = os.getenv("OGX_CORE_VLLM_EMBEDDING_TLS_VERIFY", "true")

OGX_CORE_AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "dummy")
OGX_CORE_AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "dummy")

# Remote Gemini inference provider (remote::gemini) configuration.
# The provider activates conditionally in the Red Hat OGX Distribution config.yaml
# via the pattern ${env.GEMINI_API_KEY:+gemini-inference}; injecting a non-empty
# GEMINI_API_KEY into the OgxServer pod is what turns the provider on.
GEMINI_PROVIDER_TYPE = "remote::gemini"
# provider_id reported by GET /v1/providers for the Gemini inference provider.
GEMINI_PROVIDER_ID = os.getenv("OGX_CORE_GEMINI_PROVIDER_ID", "gemini")
# Header used to override provider configuration (e.g. the API key) per request.
GEMINI_PROVIDER_DATA_HEADER = "x-ogx-provider-data"

# Primary API key injected into the distribution via the ogx-distribution-secret.
GEMINI_API_KEY = os.getenv("OGX_CORE_GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
# Secondary valid keys used by per-request-override / multi-tenant test cases.
GEMINI_API_KEY_SECONDARY = os.getenv("OGX_CORE_GEMINI_API_KEY_SECONDARY", "")
GEMINI_API_KEY_SECONDARY_2 = os.getenv("OGX_CORE_GEMINI_API_KEY_SECONDARY_2", "")

# Optional explicit model ids. When empty, tests resolve the Gemini model
# dynamically from GET /v1/models by filtering on the Gemini provider_id.
GEMINI_INFERENCE_MODEL = os.getenv("OGX_CORE_GEMINI_INFERENCE_MODEL", "")
GEMINI_EMBEDDING_MODEL = os.getenv("OGX_CORE_GEMINI_EMBEDDING_MODEL", "")

OGX_SERVER_SECRET_DATA = {
    "postgres-user": POSTGRESQL_USER,
    "postgres-password": POSTGRESQL_PASSWORD,
    "vllm-api-token": OGX_CORE_VLLM_API_TOKEN,
    "vllm-embedding-api-token": OGX_CORE_VLLM_EMBEDDING_API_TOKEN,
    "aws-access-key-id": OGX_CORE_AWS_ACCESS_KEY_ID,
    "aws-secret-access-key": OGX_CORE_AWS_SECRET_ACCESS_KEY,
    "gemini-api-token": GEMINI_API_KEY,
}

UPGRADE_DISTRIBUTION_NAME = "ogx-server-upgrade"

FAITHFULNESS_THRESHOLD = 0.5
ANSWER_RELEVANCY_THRESHOLD = 0.5
CONTEXT_PRECISION_THRESHOLD = 0.5
CONTEXT_RECALL_THRESHOLD = 0.5

_ragas_max_samples_raw = os.getenv("RAGAS_MAX_SAMPLES", "5")
try:
    RAGAS_MAX_SAMPLES = int(_ragas_max_samples_raw)
except ValueError:
    RAGAS_MAX_SAMPLES = 5

RAGAS_EVAL_MAX_TOKENS = 16384
