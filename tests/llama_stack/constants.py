from enum import Enum


class LlamaStackProviders:
    """LlamaStack provider identifiers."""

    class Inference(str, Enum):
        VLLM_INFERENCE = "vllm-inference"

    class Safety(str, Enum):
        TRUSTYAI_FMS = "trustyai_fms"

    class Eval(str, Enum):
        TRUSTYAI_LMEVAL = "trustyai_lmeval"
