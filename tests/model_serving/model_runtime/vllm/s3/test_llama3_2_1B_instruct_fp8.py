# noqa: N999
from collections.abc import Generator
from typing import Any

import pytest
import structlog
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_runtime.vllm.constant import (
    BASE_RAW_DEPLOYMENT_CONFIG,
    CHAT_QUERY,
    COMPLETION_QUERY,
)
from tests.model_serving.model_runtime.vllm.utils import validate_raw_openai_inference_request
from utilities.constants import KServeDeploymentType

LOGGER = structlog.get_logger(name=__name__)

# Llama-3.2-1B-Instruct-FP8 is a pre-quantized checkpoint: its quantization_config
# lives in the model's config.json, so vLLM picks FP8 up on its own. Passing
# --quantization would be redundant, and fp8 is not in VLLM_SUPPORTED_QUANTIZATION
# anyway. dtype stays auto so the checkpoint's own dtype is honoured.
SERVING_ARGUMENT: list[str] = [
    "--model=/mnt/models",
    "--uvicorn-log-level=debug",
    "--dtype=auto",
]

MODEL_PATH: str = "Llama-3.2-1B-Instruct-FP8"

pytestmark = pytest.mark.usefixtures("skip_if_no_supported_accelerator_type", "valid_aws_config")


@pytest.mark.vllm_nvidia_arm_gpu
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, serving_runtime, vllm_inference_service",
    [
        pytest.param(
            {"name": "llama-32-1b-fp8-standard"},
            {"model-dir": MODEL_PATH},
            {"deployment_mode": KServeDeploymentType.STANDARD},
            {
                **BASE_RAW_DEPLOYMENT_CONFIG,
                "runtime_argument": SERVING_ARGUMENT,
                "gpu_count": 1,
                "name": "llama-32-1b-fp8-standard",
            },
            id="test_llama_32_1b_instruct_fp8_standard_single_gpu",
        ),
    ],
    indirect=True,
)
class TestLlama321BInstructFP8Model:
    """Deploy the FP8-quantized Llama-3.2-1B-Instruct on a CUDA GPU and verify OpenAI inference."""

    def test_llama3_2_1b_instruct_fp8_raw_simple_openai_model_inference(
        self,
        vllm_inference_service: Generator[InferenceService, Any, Any],
        skip_if_not_raw_deployment: Any,
        response_snapshot: Any,
    ):
        """Test steps:

        Given a vLLM CUDA ServingRuntime and an FP8 Llama-3.2-1B-Instruct backed by S3 storage
        When chat completions and completions requests are sent to the OpenAI endpoints
        Then every response is successful and the generated text matches the expected keywords
        """
        validate_raw_openai_inference_request(
            isvc=vllm_inference_service,
            response_snapshot=response_snapshot,
            chat_query=CHAT_QUERY,
            completion_query=COMPLETION_QUERY,
        )
