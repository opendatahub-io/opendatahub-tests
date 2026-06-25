import os
from typing import NamedTuple

import semver
from ogx_client.types import Model
from semver import VersionInfo


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

POSTGRES_IMAGE = os.getenv(
    "OGX_VECTOR_IO_POSTGRES_IMAGE",
    (
        "registry.redhat.io/rhel9/postgresql-15@sha256:"
        "90ec347a35ab8a5d530c8d09f5347b13cc71df04f3b994bfa8b1a409b1171d59"  # postgres 15 # pragma: allowlist secret
    ),
)
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

OGX_SERVER_SECRET_DATA = {
    "postgres-user": POSTGRESQL_USER,
    "postgres-password": POSTGRESQL_PASSWORD,
    "vllm-api-token": OGX_CORE_VLLM_API_TOKEN,
    "vllm-embedding-api-token": OGX_CORE_VLLM_EMBEDDING_API_TOKEN,
    "aws-access-key-id": OGX_CORE_AWS_ACCESS_KEY_ID,
    "aws-secret-access-key": OGX_CORE_AWS_SECRET_ACCESS_KEY,
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
