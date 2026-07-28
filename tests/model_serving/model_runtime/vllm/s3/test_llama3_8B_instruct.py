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


SERVING_ARGUMENT: list[str] = [
    "--model=/mnt/models",
    "--uvicorn-log-level=debug",
    "--dtype=float16",
]

MODEL_PATH: str = "Meta-Llama-3-8B-Instruct"

BASE_RAW_DEPLOYMENT_CONFIG["runtime_argument"] = SERVING_ARGUMENT

pytestmark = pytest.mark.usefixtures("skip_if_no_supported_accelerator_type", "valid_aws_config")


@pytest.mark.vllm_nvidia_single_gpu
@pytest.mark.vllm_amd_gpu
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, serving_runtime, vllm_inference_service",
    [
        pytest.param(
            {"name": "llama-instruct-8b-standard"},
            {"model-dir": MODEL_PATH},
            {"deployment_mode": KServeDeploymentType.STANDARD},
            {
                **BASE_RAW_DEPLOYMENT_CONFIG,
                "gpu_count": 1,
                "name": "llama-instruct-standard",
            },
            id="llama-instruct-8b-standard-single-gpu",
        ),
    ],
    indirect=True,
)
class TestLlamaInstructModel:
    def test_llama3_instruct_8b_raw_simple_openai_model_inference(
        self,
        vllm_inference_service: Generator[InferenceService, Any, Any],
        skip_if_not_raw_deployment: Any,
        response_snapshot: Any,
    ):
        validate_raw_openai_inference_request(
            isvc=vllm_inference_service,
            response_snapshot=response_snapshot,
            chat_query=CHAT_QUERY,
            completion_query=COMPLETION_QUERY,
        )
