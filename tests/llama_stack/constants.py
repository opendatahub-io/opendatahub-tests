import os
from enum import Enum
from typing import NamedTuple

import semver
from llama_stack_client.types import Model
from semver import VersionInfo


class LlamaStackProviders:
    """LlamaStack provider identifiers."""

    class Inference(str, Enum):
        VLLM_INFERENCE = "vllm-inference"

    class Eval(str, Enum):
        TRUSTYAI_LMEVAL = "trustyai_lmeval"


class ModelInfo(NamedTuple):
    """Container for model information from LlamaStack client."""

    model_id: str
    embedding_model: Model
    embedding_dimension: int  # API returns integer (e.g., 768)


HTTPS_PROXY: str = os.getenv("SQUID_HTTPS_PROXY", "")

# LLS_CLIENT_VERIFY_SSL is false by default to be able to test with Self-Signed certificates
LLS_CLIENT_VERIFY_SSL = os.getenv("LLS_CLIENT_VERIFY_SSL", "false").lower() == "true"
LLS_CORE_POD_FILTER: str = "app=llama-stack"
LLS_OPENSHIFT_MINIMAL_VERSION: VersionInfo = semver.VersionInfo.parse("4.17.0")

POSTGRES_IMAGE = os.getenv(
    "LLS_VECTOR_IO_POSTGRES_IMAGE",
    (
        "registry.redhat.io/rhel9/postgresql-15@sha256:"
        "90ec347a35ab8a5d530c8d09f5347b13cc71df04f3b994bfa8b1a409b1171d59"  # postgres 15 # pragma: allowlist secret
    ),
)
POSTGRESQL_USER = os.getenv("LLS_VECTOR_IO_POSTGRESQL_USER", "ps_user")
POSTGRESQL_PASSWORD = os.getenv("LLS_VECTOR_IO_POSTGRESQL_PASSWORD", "ps_password")

LLS_CORE_INFERENCE_MODEL = os.getenv("LLS_CORE_INFERENCE_MODEL", "")
LLS_CORE_VLLM_URL = os.getenv("LLS_CORE_VLLM_URL", "")
LLS_CORE_VLLM_API_TOKEN = os.getenv("LLS_CORE_VLLM_API_TOKEN", "")
LLS_CORE_VLLM_MAX_TOKENS = os.getenv("LLS_CORE_VLLM_MAX_TOKENS", "16384")
LLS_CORE_VLLM_TLS_VERIFY = os.getenv("LLS_CORE_VLLM_TLS_VERIFY", "true")

LLS_CORE_EMBEDDING_MODEL = os.getenv("LLS_CORE_EMBEDDING_MODEL", "nomic-embed-text-v1-5")
LLS_CORE_EMBEDDING_PROVIDER_MODEL_ID = os.getenv("LLS_CORE_EMBEDDING_PROVIDER_MODEL_ID", "nomic-embed-text-v1-5")
LLS_CORE_VLLM_EMBEDDING_URL = os.getenv(
    "LLS_CORE_VLLM_EMBEDDING_URL", "https://nomic-embed-text-v1-5.example.com:443/v1"
)
LLS_CORE_VLLM_EMBEDDING_API_TOKEN = os.getenv("LLS_CORE_VLLM_EMBEDDING_API_TOKEN", "fake")
LLS_CORE_VLLM_EMBEDDING_MAX_TOKENS = os.getenv("LLS_CORE_VLLM_EMBEDDING_MAX_TOKENS", "8192")
LLS_CORE_VLLM_EMBEDDING_TLS_VERIFY = os.getenv("LLS_CORE_VLLM_EMBEDDING_TLS_VERIFY", "true")

LLS_CORE_AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
LLS_CORE_AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")

LLAMA_STACK_DISTRIBUTION_SECRET_DATA = {
    "postgres-user": POSTGRESQL_USER,
    "postgres-password": POSTGRESQL_PASSWORD,
    "vllm-api-token": LLS_CORE_VLLM_API_TOKEN,
    "vllm-embedding-api-token": LLS_CORE_VLLM_EMBEDDING_API_TOKEN,
    "aws-access-key-id": LLS_CORE_AWS_ACCESS_KEY_ID,
    "aws-secret-access-key": LLS_CORE_AWS_SECRET_ACCESS_KEY,
}

UPGRADE_DISTRIBUTION_NAME = "llama-stack-distribution-upgrade"
