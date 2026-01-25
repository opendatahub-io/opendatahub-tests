from dataclasses import dataclass
from enum import Enum
from typing import List, NamedTuple, TypedDict
from llama_stack_client.types import Model
from semver import VersionInfo
import semver


class LlamaStackProviders:
    """LlamaStack provider identifiers."""

    class Inference(str, Enum):
        VLLM_INFERENCE = "vllm-inference"

    class Safety(str, Enum):
        TRUSTYAI_FMS = "trustyai_fms"

    class Eval(str, Enum):
        TRUSTYAI_LMEVAL = "trustyai_lmeval"
        TRUSTYAI_RAGAS_INLINE = "trustyai_ragas_inline"
        TRUSTYAI_RAGAS_REMOTE = "trustyai_ragas_remote"


class ModelInfo(NamedTuple):
    """Container for model information from LlamaStack client."""

    model_id: str
    embedding_model: Model
    embedding_dimension: int  # API returns integer (e.g., 768)


LLS_CORE_POD_FILTER: str = "app=llama-stack"
LLS_OPENSHIFT_MINIMAL_VERSION: VersionInfo = semver.VersionInfo.parse("4.17.0")


class TurnExpectation(TypedDict):
    question: str
    expected_keywords: List[str]
    description: str


class TurnResult(TypedDict):
    question: str
    description: str
    expected_keywords: List[str]
    found_keywords: List[str]
    missing_keywords: List[str]
    response_content: str
    response_length: int
    event_count: int
    success: bool
    error: str | None


class ValidationSummary(TypedDict):
    total_turns: int
    successful_turns: int
    failed_turns: int
    success_rate: float
    total_events: int
    total_response_length: int


class ValidationResult(TypedDict):
    success: bool
    results: List[TurnResult]
    summary: ValidationSummary


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
