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
