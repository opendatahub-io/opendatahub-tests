from dataclasses import dataclass
from enum import Enum
from typing import List


class LlamaStackProviders:
    """LlamaStack provider identifiers."""

    class Inference(str, Enum):
        VLLM_INFERENCE = "vllm-inference"

    class Safety(str, Enum):
        TRUSTYAI_FMS = "trustyai_fms"

    class Eval(str, Enum):
        TRUSTYAI_LMEVAL = "trustyai_lmeval"


@dataclass
class TorchTuneTestExpectation:
    """Test expectation for TorchTune documentation questions."""

    question: str
    expected_keywords: List[str]
    description: str


TORCHTUNE_TEST_EXPECTATIONS: List[TorchTuneTestExpectation] = [
    TorchTuneTestExpectation(
        question="what is torchtune",
        expected_keywords=["torchtune", "pytorch", "fine-tuning", "training", "model"],
        description="Should provide information about torchtune framework",
    ),
    TorchTuneTestExpectation(
        question="What do you know about LoRA?",
        expected_keywords=[
            "LoRA",
            "parameter",
            "efficient",
            "fine-tuning",
            "reduce",
        ],
        description="Should provide information about LoRA (Low Rank Adaptation)",
    ),
    TorchTuneTestExpectation(
        question="How can I optimize model training for quantization?",
        expected_keywords=[
            "Quantization-Aware Training",
            "QAT",
            "training",
            "fine-tuning",
            "fake",
            "quantized",
        ],
        description="Should provide information about QAT (Quantization-Aware Training)",
    ),
    TorchTuneTestExpectation(
        question="Are there any memory optimizations for LoRA?",
        expected_keywords=["QLoRA", "fine-tuning", "4-bit", "Optimization", "LoRA"],
        description="Should provide information about QLoRA",
    ),
    TorchTuneTestExpectation(
        question="tell me about dora",
        expected_keywords=["dora", "parameter", "magnitude", "direction", "fine-tuning"],
        description="Should provide information about DoRA (Weight-Decomposed Low-Rank Adaptation)",
    ),
]
