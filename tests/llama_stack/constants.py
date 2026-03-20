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

    class Safety(str, Enum):
        TRUSTYAI_FMS = "trustyai_fms"

    class Eval(str, Enum):
        TRUSTYAI_LMEVAL = "trustyai_lmeval"


class ModelInfo(NamedTuple):
    """Container for model information from LlamaStack client."""

    model_id: str
    embedding_model: Model
    embedding_dimension: int  # API returns integer (e.g., 768)


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

FINANCE_DOCS = "tests/llama_stack/dataset/corpus/finance"
IBM_2025_Q4_EARNINGS_DOC_ENCRYPTED = "tests/llama_stack/dataset/corpus/pdf-testing/ibm-4q25-press-release-encrypted.pdf"
IBM_2025_Q4_EARNINGS_DOC_UNENCRYPTED = (
    "tests/llama_stack/dataset/corpus/finance/ibm-4q25-earnings-press-release-unencrypted.pdf"
)
IBM_EARNINGS_SEARCH_QUERIES_BY_MODE: dict[str, list[str]] = {
    "vector": [
        "How did IBM perform financially in the fourth quarter of 2025?",
        "What were the main drivers of revenue growth?",
        "What is the company outlook for 2026?",
        "How did profit margins change year over year?",
        "What did leadership say about generative AI and growth?",
    ],
    "keyword": [
        "What was free cash flow in the fourth quarter?",
        "What was Consulting revenue and segment profit margin?",
        "What was Software revenue and constant currency growth?",
        "What was diluted earnings per share for continuing operations?",
        "What are full-year 2026 expectations for revenue and free cash flow?",
    ],
    "hybrid": [
        "What was IBM free cash flow and what does the company expect for 2026?",
        "What were segment results for Software and Infrastructure revenue?",
        "What was GAAP gross profit margin and pre-tax income?",
        "What did James Kavanaugh say about 2025 results and 2026 prospects?",
        "What was Consulting revenue and segment profit margin?",
    ],
}
