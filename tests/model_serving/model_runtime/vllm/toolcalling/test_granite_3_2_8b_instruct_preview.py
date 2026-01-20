import pytest
from simple_logger.logger import get_logger
from typing import Any, Generator
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from utilities.constants import KServeDeploymentType
from tests.model_serving.model_runtime.vllm.utils import (
    validate_raw_openai_inference_request,
    validate_raw_tgis_inference_request,
)
from tests.model_serving.model_runtime.vllm.constant import (
    LIGHTSPEED_TOOL_QUERY,
    LIGHTSPEED_TOOL,
    WEATHER_TOOL,
    WEATHER_TOOL_QUERY,
    MATH_CHAT_QUERY,
    COMPLETION_QUERY,
)

LOGGER = get_logger(name=__name__)


SERVING_ARGUMENT: list[str] = [
    "--model=/mnt/models",
    "--uvicorn-log-level=debug",
    "--dtype=float16",
    "--chat-template=/opt/app-root/template/tool_chat_template_granite.jinja",
    "--enable-auto-tool-choice",
    "--tool-call-parser=granite",
]

MODEL_PATH: str = "ibm-granite/granite-3.2-8b-instruct-preview"


BASE_DEPLOYMENT_CONFIG: dict[str, Any] = {
    "deployment_mode": KServeDeploymentType.RAW_DEPLOYMENT,
    "runtime_argument": SERVING_ARGUMENT,
    "min-replicas": 1,
}

pytestmark = pytest.mark.usefixtures("skip_if_no_supported_accelerator_type", "valid_aws_config")


@pytest.mark.vllm_nvidia_single_gpu
@pytest.mark.vllm_amd_gpu
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, serving_runtime, vllm_inference_service",
    [
        pytest.param(
            {"name": "granite-32-8b"},
            {"model-dir": MODEL_PATH},
            {"deployment_type": KServeDeploymentType.RAW_DEPLOYMENT},
            {
                **BASE_DEPLOYMENT_CONFIG,
                "gpu_count": 1,
                "name": "granite-32-8b",
            },
        ),
    ],
    indirect=True,
)
class TestGranite32ToolModel:
    def test_granite_simple_openai_model_inference(
        self,
        vllm_inference_service: Generator[InferenceService, Any, Any],
        vllm_pod_resource: Pod,
        response_snapshot: Any,
    ):
        validate_raw_openai_inference_request(
            pod_name=vllm_pod_resource.name,
            isvc=vllm_inference_service,
            response_snapshot=response_snapshot,
            chat_query=MATH_CHAT_QUERY,
            completion_query=COMPLETION_QUERY,
        )

    def test_granite_simple_tgis_model_inference(
        self,
        vllm_inference_service: Generator[InferenceService, Any, Any],
        vllm_pod_resource: Pod,
        response_snapshot: Any,
    ):
        validate_raw_tgis_inference_request(
            pod_name=vllm_pod_resource.name,
            isvc=vllm_inference_service,
            response_snapshot=response_snapshot,
            completion_query=COMPLETION_QUERY,
        )

    def test_granite_model_lightspeed_tool_inference(
        self,
        vllm_inference_service: Generator[InferenceService, Any, Any],
        vllm_pod_resource: Pod,
        response_snapshot: Any,
    ):
        validate_raw_openai_inference_request(
            pod_name=vllm_pod_resource.name,
            isvc=vllm_inference_service,
            response_snapshot=response_snapshot,
            chat_query=LIGHTSPEED_TOOL_QUERY,
            completion_query=COMPLETION_QUERY,
            tool_calling=LIGHTSPEED_TOOL[0],
        )

    def test_granite_model_weather_tool_inference(
        self,
        vllm_inference_service: Generator[InferenceService, Any, Any],
        vllm_pod_resource: Pod,
        response_snapshot: Any,
    ):
        validate_raw_openai_inference_request(
            pod_name=vllm_pod_resource.name,
            isvc=vllm_inference_service,
            response_snapshot=response_snapshot,
            chat_query=WEATHER_TOOL_QUERY,
            completion_query=COMPLETION_QUERY,
            tool_calling=WEATHER_TOOL[0],
        )


@pytest.mark.vllm_nvidia_multi_gpu
@pytest.mark.vllm_amd_gpu
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, serving_runtime, vllm_inference_service",
    [
        pytest.param(
            {"name": "granite-32-8b-multi"},
            {"model-dir": MODEL_PATH},
            {"deployment_type": KServeDeploymentType.RAW_DEPLOYMENT},
            {
                **BASE_DEPLOYMENT_CONFIG,
                "gpu_count": 2,
                "name": "granite-32-8b-multi",
            },
        ),
    ],
    indirect=True,
)
class TestGranite32ToolMultiModel:
    def test_granite_multi_openai_model_inference(
        self,
        vllm_inference_service: Generator[InferenceService, Any, Any],
        vllm_pod_resource: Pod,
        response_snapshot: Any,
    ):
        validate_raw_openai_inference_request(
            pod_name=vllm_pod_resource.name,
            isvc=vllm_inference_service,
            response_snapshot=response_snapshot,
            chat_query=MATH_CHAT_QUERY,
            completion_query=COMPLETION_QUERY,
        )

    def test_granite_multi_tgis_model_inference(
        self,
        vllm_inference_service: Generator[InferenceService, Any, Any],
        vllm_pod_resource: Pod,
        response_snapshot: Any,
    ):
        validate_raw_tgis_inference_request(
            pod_name=vllm_pod_resource.name,
            isvc=vllm_inference_service,
            response_snapshot=response_snapshot,
            completion_query=COMPLETION_QUERY,
        )

    def test_granite_multi_model_lightspeed_tool_inference(
        self,
        vllm_inference_service: Generator[InferenceService, Any, Any],
        vllm_pod_resource: Pod,
        response_snapshot: Any,
    ):
        validate_raw_openai_inference_request(
            pod_name=vllm_pod_resource.name,
            isvc=vllm_inference_service,
            response_snapshot=response_snapshot,
            chat_query=LIGHTSPEED_TOOL_QUERY,
            completion_query=COMPLETION_QUERY,
            tool_calling=LIGHTSPEED_TOOL[0],
        )

    def test_granite_multi_model_weather_tool_inference(
        self,
        vllm_inference_service: Generator[InferenceService, Any, Any],
        vllm_pod_resource: Pod,
        response_snapshot: Any,
    ):
        validate_raw_openai_inference_request(
            pod_name=vllm_pod_resource.name,
            isvc=vllm_inference_service,
            response_snapshot=response_snapshot,
            chat_query=WEATHER_TOOL_QUERY,
            completion_query=COMPLETION_QUERY,
            tool_calling=WEATHER_TOOL[0],
        )
