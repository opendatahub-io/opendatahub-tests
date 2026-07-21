from collections.abc import Generator
from typing import Any

import pytest
import structlog
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_runtime.vllm.constant import BASE_RAW_DEPLOYMENT_CONFIG
from tests.model_serving.model_runtime.vllm.cpu.ibm_power_z.constant import (
    GRANITE_3B_CODE_INSTRUCT_2K_MODEL_PATH,
    IBM_POWER_Z_CHAT_INFERENCE_REQUEST,
    IBM_POWER_Z_MODEL_ENV_VARIABLES,
    IBM_POWER_Z_SERVING_ARGUMENT,
)
from tests.model_serving.model_runtime.vllm.cpu.ibm_power_z.utils import validate_ibm_power_z_chat_completions_request
from utilities.constants import KServeDeploymentType

LOGGER = structlog.get_logger(name=__name__)

pytestmark = pytest.mark.usefixtures("skip_if_no_supported_ibm_power_z_accelerator_type", "valid_aws_config")


@pytest.mark.vllm_cpu_power
@pytest.mark.vllm_cpu_z
@pytest.mark.parametrize(
    (
        "model_namespace",
        "s3_models_storage_uri",
        "ibm_power_z_serving_runtime",
        "ibm_power_z_inference_service",
        "inference_request",
    ),
    [
        pytest.param(
            {"name": "granite-3b-code-2k-cpu"},
            {"model-dir": GRANITE_3B_CODE_INSTRUCT_2K_MODEL_PATH},
            {"deployment_mode": KServeDeploymentType.STANDARD},
            {
                **BASE_RAW_DEPLOYMENT_CONFIG,
                "name": "granite-3b-code-2k-cpu",
                "runtime_argument": IBM_POWER_Z_SERVING_ARGUMENT,
                "model_env_variables": IBM_POWER_Z_MODEL_ENV_VARIABLES,
            },
            IBM_POWER_Z_CHAT_INFERENCE_REQUEST,
            id="test_granite_3b_code_2k_cpu",
        ),
    ],
    indirect=[
        "model_namespace",
        "s3_models_storage_uri",
        "ibm_power_z_serving_runtime",
        "ibm_power_z_inference_service",
    ],
)
class TestGranite3BCodeInstruct2K:
    """Deploy Granite-3B-Code-Instruct-2K on IBM Power or Z and verify chat completions inference."""

    def test_granite_3b_code_instruct_2k_chat_inference(
        self,
        ibm_power_z_inference_service: Generator[InferenceService, Any, Any],
        skip_if_not_ibm_power_z_raw_deployment: Any,
        inference_request: dict[str, Any],
    ):
        """Test steps:

        Given a vLLM CPU ServingRuntime and Granite-3B-Code-Instruct-2K backed by S3 storage
        When a POST request is sent to /v1/chat/completions
        Then the response status is 200 and the completion text is non-empty
        """
        validate_ibm_power_z_chat_completions_request(
            isvc=ibm_power_z_inference_service,
            inference_request=inference_request,
        )
