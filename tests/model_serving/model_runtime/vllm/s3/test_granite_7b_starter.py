from collections.abc import Generator
from typing import Any

import pytest
import structlog
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_runtime.vllm.constant import (
    BASE_RAW_DEPLOYMENT_CONFIG,
    COMPLETION_QUERY,
    GRANITE_CHAT_QUERY,
    GRANITE_SERVING_ARGUMENT,
)
from tests.model_serving.model_runtime.vllm.utils import validate_raw_openai_inference_request
from utilities.constants import KServeDeploymentType

LOGGER = structlog.get_logger(name=__name__)


MODEL_PATH: str = "granite-7b-starter"

BASE_RAW_DEPLOYMENT_CONFIG["runtime_argument"] = GRANITE_SERVING_ARGUMENT

pytestmark = pytest.mark.usefixtures("skip_if_no_supported_accelerator_type", "valid_aws_config")


@pytest.mark.vllm_nvidia_multi_gpu
@pytest.mark.vllm_amd_gpu
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, serving_runtime, vllm_inference_service",
    [
        pytest.param(
            {"name": "granite-starter-standard"},
            {"model-dir": MODEL_PATH},
            {"deployment_mode": KServeDeploymentType.STANDARD},
            {
                **BASE_RAW_DEPLOYMENT_CONFIG,
                "gpu_count": 2,
                "name": "granite-starter-standard",
            },
            id="granite-starter-standard-multi-gpu",
        ),
    ],
    indirect=True,
)
class TestGraniteStarterModel:
    def test_granite_starter_raw_simple_openai_model_inference(
        self,
        vllm_inference_service: Generator[InferenceService, Any, Any],
        skip_if_not_raw_deployment: Any,
        response_snapshot: Any,
    ):
        validate_raw_openai_inference_request(
            isvc=vllm_inference_service,
            response_snapshot=response_snapshot,
            chat_query=GRANITE_CHAT_QUERY,
            completion_query=COMPLETION_QUERY,
        )
