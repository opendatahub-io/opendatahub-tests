from collections.abc import Generator
from typing import Any

import pytest
import structlog
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_runtime.vllm.constant import BASE_RAW_DEPLOYMENT_CONFIG
from tests.model_serving.model_runtime.vllm.cpu.cpu_x86.constant import (
    CPU_X86_ENV_VARIABLES,
    CPU_X86_SERVING_ARGUMENT,
    OPT_125M_COMPLETION_REQUEST,
    OPT_125M_MODEL_PATH,
    TINYLLAMA_CHAT_COMPLETION_REQUEST,
    TINYLLAMA_MODEL_PATH,
)
from tests.model_serving.model_runtime.vllm.cpu.cpu_x86.utils import validate_cpu_x86_inference_request
from utilities.constants import KServeDeploymentType

LOGGER = structlog.get_logger(name=__name__)

pytestmark = pytest.mark.usefixtures("skip_if_no_supported_cpu_x86_accelerator_type", "valid_aws_config")


@pytest.mark.smoke
@pytest.mark.vllm_cpu_x86
@pytest.mark.parametrize(
    (
        "model_namespace",
        "s3_models_storage_uri",
        "cpu_x86_serving_runtime",
        "cpu_x86_inference_service",
        "inference_request",
    ),
    [
        pytest.param(
            {"name": "opt-125m-raw-cpu"},
            {"model-dir": OPT_125M_MODEL_PATH},
            {"deployment_mode": KServeDeploymentType.RAW_DEPLOYMENT},
            {
                **BASE_RAW_DEPLOYMENT_CONFIG,
                "name": "opt-125m-raw-cpu",
                "runtime_argument": CPU_X86_SERVING_ARGUMENT,
                "model_env_variables": CPU_X86_ENV_VARIABLES,
            },
            OPT_125M_COMPLETION_REQUEST,
            id="facebook-opt-125m-raw-cpu",
            marks=[pytest.mark.smoke],
        ),
        pytest.param(
            {"name": "tinyllama-raw-cpu"},
            {"model-dir": TINYLLAMA_MODEL_PATH},
            {"deployment_mode": KServeDeploymentType.RAW_DEPLOYMENT},
            {
                **BASE_RAW_DEPLOYMENT_CONFIG,
                "name": "tinyllama-raw-cpu",
                "runtime_argument": CPU_X86_SERVING_ARGUMENT,
                "model_env_variables": CPU_X86_ENV_VARIABLES,
            },
            TINYLLAMA_CHAT_COMPLETION_REQUEST,
            id="tinyllama-1-1b-chat-raw-cpu",
            marks=[pytest.mark.tier1],
        ),
    ],
    indirect=["model_namespace", "s3_models_storage_uri", "cpu_x86_serving_runtime", "cpu_x86_inference_service"],
)
class TestVllmCpuX86S3Inference:
    """Deploy vLLM CPU x86 models from S3 and verify OpenAI inference."""

    def test_vllm_cpu_x86_inference(
        self,
        cpu_x86_inference_service: Generator[InferenceService, Any, Any],
        skip_if_not_cpu_x86_raw_deployment: Any,
        inference_request: dict[str, Any],
    ):
        """Test steps:

        Given a vLLM CPU x86 ServingRuntime and an InferenceService backed by S3 storage
        When a POST request is sent to the model-appropriate OpenAI endpoint
        Then the response status is 200 and the completion text is non-empty
        """
        validate_cpu_x86_inference_request(
            isvc=cpu_x86_inference_service,
            inference_request=inference_request,
        )
