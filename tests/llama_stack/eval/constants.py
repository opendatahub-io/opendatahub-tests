import os

from tests.llama_stack.constants import LlamaStackProviders
from utilities.constants import QWEN_MODEL_NAME

DK_CUSTOM_DATASET_IMAGE: str = (
    "quay.io/trustyai_testing/dkbench-dataset@sha256:27e8946c878b52a72c272c27862ee1f0384d06a9b6985251c011dfe6ae8bf23d"
)

LLAMA_STACK_DISTRIBUTION_IMAGE: str = os.getenv(
    "LLS_DISTRIBUTION_IMAGE",
    "quay.io/opendatahub/llama-stack@sha256:cf21d3919d265f8796ed600bfe3d2eb3ce797b35ab8e60ca9b6867e0516675e5",
)

# Garak benchmark IDs
GARAK_QUICK_BENCHMARK_ID: str = "trustyai_garak::quick"
GARAK_CUSTOM_BENCHMARK_ID: str = "garak_custom_prompt_injection"
GARAK_SHIELD_BENCHMARK_ID: str = "garak_shield_scan"

# Garak timeouts (seconds)
GARAK_DEFAULT_TIMEOUT: int = 900
GARAK_SHIELD_TIMEOUT: int = 1200

# LlamaStack 0.5.x qualifies model IDs with the provider prefix
QWEN_LLAMA_STACK_MODEL_ID: str = f"{LlamaStackProviders.Inference.VLLM_INFERENCE.value}/{QWEN_MODEL_NAME}"

# Garak shield config
GARAK_SHIELD_ID: str = "content_safety"
