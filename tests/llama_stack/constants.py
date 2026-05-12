import os
from enum import Enum
from typing import NamedTuple

import semver
from llama_stack_client.types import Model
from semver import VersionInfo

from utilities.llama_stack_utils import (
    LLAMA_STACK_DISTRIBUTION_SECRET_DATA,
    LLS_CLIENT_VERIFY_SSL,
    LLS_CORE_POD_FILTER,
    POSTGRES_IMAGE,
)


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

LLS_OPENSHIFT_MINIMAL_VERSION: VersionInfo = semver.VersionInfo.parse("4.17.0")

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

UPGRADE_DISTRIBUTION_NAME = "llama-stack-distribution-upgrade"

FAITHFULNESS_THRESHOLD = 0.5
ANSWER_RELEVANCY_THRESHOLD = 0.5
CONTEXT_PRECISION_THRESHOLD = 0.5
CONTEXT_RECALL_THRESHOLD = 0.5

_ragas_max_samples_raw = os.getenv("RAGAS_MAX_SAMPLES", "5")
try:
    RAGAS_MAX_SAMPLES = int(_ragas_max_samples_raw)
except ValueError:
    RAGAS_MAX_SAMPLES = 5
    
__all__ = [
    "LLAMA_STACK_DISTRIBUTION_SECRET_DATA",
    "LLS_CLIENT_VERIFY_SSL",
    "LLS_CORE_POD_FILTER",
    "POSTGRES_IMAGE",
]
