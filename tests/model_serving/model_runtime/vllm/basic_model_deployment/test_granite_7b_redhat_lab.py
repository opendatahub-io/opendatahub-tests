from collections.abc import Generator
from typing import Any

import pytest
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from simple_logger.logger import get_logger

from tests.model_serving.model_runtime.vllm.constant import (
    BASE_RAW_DEPLOYMENT_CONFIG,
    CHAT_QUERY,
    COMPLETION_QUERY,
)
from tests.model_serving.model_runtime.vllm.utils import (
    validate_raw_openai_inference_request,
    validate_raw_tgis_inference_request,
)
from utilities.constants import KServeDeploymentType

LOGGER = get_logger(name=__name__)


SERVING_ARGUMENT: list[str] = [
    "--model=/mnt/models",
    "--uvicorn-log-level=debug",
    "--dtype=float16",
    "--chat-template=/opt/app-root/template/template_chatglm.jinja",
]

MODEL_PATH: str = "granite-7b-redhat-lab"

BASE_RAW_DEPLOYMENT_CONFIG["runtime_argument"] = SERVING_ARGUMENT

pytestmark = pytest.mark.usefixtures("skip_if_no_supported_accelerator_type", "valid_aws_config")


@pytest.mark.vllm_nvidia_single_gpu
@pytest.mark.vllm_amd_gpu
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, serving_runtime, vllm_inference_service",
    [
        pytest.param(
            {"name": "granite-lab-raw"},
            {"model-dir": MODEL_PATH},
            {"deployment_type": KServeDeploymentType.RAW_DEPLOYMENT},
            {
                **BASE_RAW_DEPLOYMENT_CONFIG,
                "gpu_count": 1,
                "name": "granite-lab-raw",
            },
            id="granite-lab-raw-single-gpu",
        ),
    ],
    indirect=True,
)
class TestGraniteLabModel:
    def test_granite_lab_raw_simple_openai_model_inference(
        self,
        vllm_inference_service: Generator[InferenceService, Any, Any],
        skip_if_not_raw_deployment: Any,
        vllm_pod_resource: Pod,
        response_snapshot: Any,
    ):
        validate_raw_openai_inference_request(
            pod_name=vllm_pod_resource.name,
            isvc=vllm_inference_service,
            response_snapshot=response_snapshot,
            chat_query=CHAT_QUERY,
            completion_query=COMPLETION_QUERY,
        )

    def test_granite_lab_raw_simple_tgis_model_inference(
        self,
        vllm_inference_service: Generator[InferenceService, Any, Any],
        skip_if_not_raw_deployment: Any,
        vllm_pod_resource: Pod,
        response_snapshot: Any,
    ):
        validate_raw_tgis_inference_request(
            pod_name=vllm_pod_resource.name,
            isvc=vllm_inference_service,
            response_snapshot=response_snapshot,
            completion_query=COMPLETION_QUERY,
        )


@pytest.mark.vllm_nvidia_multi_gpu
@pytest.mark.vllm_amd_gpu
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, serving_runtime, vllm_inference_service",
    [
        pytest.param(
            {"name": "granite-lab-raw-multi"},
            {"model-dir": MODEL_PATH},
            {"deployment_type": KServeDeploymentType.RAW_DEPLOYMENT},
            {
                **BASE_RAW_DEPLOYMENT_CONFIG,
                "gpu_count": 2,
                "name": "granite-lab-rm",
            },
            id="granite-lab-raw-multi-gpu",
        ),
    ],
    indirect=True,
)
class TestMultiGraniteLabModel:
    def test_granite_lab_raw_multi_openai_model_inference(
        self,
        vllm_inference_service: Generator[InferenceService, Any, Any],
        skip_if_not_raw_deployment: Any,
        vllm_pod_resource: Pod,
        response_snapshot: Any,
    ):
        validate_raw_openai_inference_request(
            pod_name=vllm_pod_resource.name,
            isvc=vllm_inference_service,
            response_snapshot=response_snapshot,
            chat_query=CHAT_QUERY,
            completion_query=COMPLETION_QUERY,
        )

    def test_granite_lab_raw_multi_tgis_model_inference(
        self,
        vllm_inference_service: Generator[InferenceService, Any, Any],
        skip_if_not_raw_deployment: Any,
        vllm_pod_resource: Pod,
        response_snapshot: Any,
    ):
        validate_raw_tgis_inference_request(
            pod_name=vllm_pod_resource.name,
            isvc=vllm_inference_service,
            response_snapshot=response_snapshot,
            completion_query=COMPLETION_QUERY,
        )
