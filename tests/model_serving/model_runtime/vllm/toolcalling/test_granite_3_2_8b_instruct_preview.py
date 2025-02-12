import pytest
from simple_logger.logger import get_logger
from typing import List, Any, Generator
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from utilities.constants import KServeDeploymentType
from tests.model_serving.model_runtime.vllm.utils import (
    run_raw_inference,
    validate_inference_output,
)
from tests.model_serving.model_runtime.vllm.constant import (
    OPENAI_ENDPOINT_NAME,
    TGIS_ENDPOINT_NAME,
    LIGHTSPEED_TOOL_QUERY,
    LIGHTSPEED_TOOL,
    WEATHER_TOOL,
    WEATHER_TOOL_QUERY,
    MATH_CHAT_QUERY,
)

LOGGER = get_logger(name=__name__)


SERVING_ARGUMENT: List[str] = [
    "--model=/mnt/models",
    "--uvicorn-log-level=debug",
    "--dtype=float16",
    "--chat-template=/app/data/template/tool_chat_template_granite.jinja",
    "--enable-auto-tool-choice",
    "--tool-call-parser=granite",
]

MODEL_PATH: str = "ibm-granite/granite-3.2-8b-instruct-preview"


pytestmark = pytest.mark.usefixtures("skip_if_no_supported_accelerator_type", "valid_aws_config")


@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, serving_runtime, vllm_inference_service",
    [
        pytest.param(
            {"name": "granite-32-8b"},
            {"model-dir": MODEL_PATH},
            {"deployment_type": KServeDeploymentType.RAW_DEPLOYMENT},
            {
                "deployment_mode": KServeDeploymentType.RAW_DEPLOYMENT,
                "runtime_argument": SERVING_ARGUMENT,
                "gpu_count": 1,
                "name": "granite-32-8b",
                "min-replicas": 1,
            },
        ),
    ],
    indirect=True,
)
class TestGranite32ToolModel:
    def test_granite_model_inference(
        self,
        vllm_inference_service: Generator[InferenceService, Any, Any],
        get_pod_name_resource: Pod,
        response_snapshot: Any,
    ):
        pod_name = get_pod_name_resource.name
        model_info, chat_responses, completion_responses = run_raw_inference(
            pod_name=pod_name,
            isvc=vllm_inference_service,
            port=8080,
            endpoint=OPENAI_ENDPOINT_NAME,
            chat_query=MATH_CHAT_QUERY,
        )
        model_info_tgis, completion_responses_tgis, completion_responses_tgis_stream = run_raw_inference(
            pod_name=pod_name,
            isvc=vllm_inference_service,
            port=8033,
            endpoint=TGIS_ENDPOINT_NAME,
        )
        validate_inference_output(
            model_info,
            chat_responses,
            completion_responses,
            model_info_tgis,
            completion_responses_tgis,
            completion_responses_tgis_stream,
            response_snapshot=response_snapshot,
        )

    def test_granite_model_tool_inference(
        self,
        vllm_inference_service: Generator[InferenceService, Any, Any],
        get_pod_name_resource: Pod,
        response_snapshot: Any,
    ):
        pod_name = get_pod_name_resource.name
        model_info_w, chat_responses_w, completion_responses_w = run_raw_inference(
            pod_name=pod_name,
            isvc=vllm_inference_service,
            port=8080,
            endpoint=OPENAI_ENDPOINT_NAME,
            chat_query=WEATHER_TOOL_QUERY,
            tool_calling=WEATHER_TOOL[0],
        )
        model_info_l, chat_responses_l, completion_responses_l = run_raw_inference(
            pod_name=pod_name,
            isvc=vllm_inference_service,
            port=8080,
            endpoint=OPENAI_ENDPOINT_NAME,
            chat_query=LIGHTSPEED_TOOL_QUERY,
            tool_calling=LIGHTSPEED_TOOL[0],
        )
        validate_inference_output(
            model_info_w,
            chat_responses_w,
            completion_responses_w,
            model_info_l,
            chat_responses_l,
            completion_responses_l,
            response_snapshot=response_snapshot,
        )


@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, serving_runtime, vllm_inference_service",
    [
        pytest.param(
            {"name": "granite-32-8b-multi"},
            {"model-dir": MODEL_PATH},
            {"deployment_type": KServeDeploymentType.RAW_DEPLOYMENT},
            {
                "deployment_mode": KServeDeploymentType.RAW_DEPLOYMENT,
                "runtime_argument": SERVING_ARGUMENT,
                "gpu_count": 2,
                "name": "granite-32-8b-multi",
                "min-replicas": 1,
            },
        ),
    ],
    indirect=True,
)
class TestGranite32ToolMultiModel:
    def test_granite_multi_model_inference(
        self,
        vllm_inference_service: Generator[InferenceService, Any, Any],
        get_pod_name_resource: Pod,
        response_snapshot: Any,
    ):
        pod_name = get_pod_name_resource.name
        model_info, chat_responses, completion_responses = run_raw_inference(
            pod_name=pod_name,
            isvc=vllm_inference_service,
            port=8080,
            endpoint=OPENAI_ENDPOINT_NAME,
        )
        model_info_tgis, completion_responses_tgis, completion_responses_tgis_stream = run_raw_inference(
            pod_name=pod_name,
            isvc=vllm_inference_service,
            port=8033,
            endpoint=TGIS_ENDPOINT_NAME,
        )
        validate_inference_output(
            model_info,
            chat_responses,
            completion_responses,
            model_info_tgis,
            completion_responses_tgis,
            completion_responses_tgis_stream,
            response_snapshot=response_snapshot,
        )

    def test_granite_multi_model_tool_inference(
        self,
        vllm_inference_service: Generator[InferenceService, Any, Any],
        get_pod_name_resource: Pod,
        response_snapshot: Any,
    ):
        pod_name = get_pod_name_resource.name
        model_info_w, chat_responses_w, completion_responses_w = run_raw_inference(
            pod_name=pod_name,
            isvc=vllm_inference_service,
            port=8080,
            endpoint=OPENAI_ENDPOINT_NAME,
            chat_query=WEATHER_TOOL_QUERY,
            tool_calling=WEATHER_TOOL[0],
        )
        model_info_l, chat_responses_l, completion_responses_l = run_raw_inference(
            pod_name=pod_name,
            isvc=vllm_inference_service,
            port=8080,
            endpoint=OPENAI_ENDPOINT_NAME,
            chat_query=LIGHTSPEED_TOOL_QUERY,
            tool_calling=LIGHTSPEED_TOOL[0],
        )
        validate_inference_output(
            model_info_w,
            chat_responses_w,
            completion_responses_w,
            model_info_l,
            chat_responses_l,
            completion_responses_l,
            response_snapshot=response_snapshot,
        )
