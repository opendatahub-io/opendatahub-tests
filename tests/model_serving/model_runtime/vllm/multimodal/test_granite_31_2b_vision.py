import pytest
from simple_logger.logger import get_logger
from utilities.constants import KServeDeploymentType
from tests.model_serving.model_runtime.vllm.utils import (
    run_raw_inference,
    validate_inference_output,
)
from tests.model_serving.model_runtime.vllm.constant import MULTI_IMAGE_QUERIES, THREE_IMAGE_QUERY

LOGGER = get_logger(name=__name__)


SERVING_ARGUMENT = ["--model=/mnt/models", "--uvicorn-log-level=debug", "--limit-mm-per-prompt", "image=2"]

MODEL_PATH = "ibm-granite/granite-vision-3.1-2b-preview"

pytestmark = pytest.mark.usefixtures("skip_if_no_supported_accelerator_type", "valid_aws_config")


@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, serving_runtime, vllm_inference_service",
    [
        pytest.param(
            {"name": "granite-2b-vision"},
            {"model-dir": MODEL_PATH},
            {"deployment_type": KServeDeploymentType.RAW_DEPLOYMENT},
            {
                "deployment_mode": KServeDeploymentType.RAW_DEPLOYMENT,
                "runtime_argument": SERVING_ARGUMENT,
                "gpu_count": 1,
                "name": "granite-2b-vision-model",
                "min-replicas": 1,
            },
        ),
    ],
    indirect=True,
)
class TestGraniteVisionModel:
    def test_single_image_inference(self, vllm_inference_service, get_pod_name_resource, response_snapshot):
        pod = get_pod_name_resource.name
        model_info, chat_responses, completion_responses = run_raw_inference(
            pod_name=pod, isvc=vllm_inference_service, port=8080, endpoint="openai", chat_query=MULTI_IMAGE_QUERIES
        )
        validate_inference_output(model_info, chat_responses, completion_responses, response_snapshot=response_snapshot)

    @pytest.mark.xfail(reason="Test expected to fail due to image limit of 2, but model query requests 3 images.")
    def test_multi_image_query_inference(self, vllm_inference_service, get_pod_name_resource, response_snapshot):
        pod = get_pod_name_resource.name
        model_info, chat_responses, completion_responses = run_raw_inference(
            pod_name=pod, isvc=vllm_inference_service, port=8080, endpoint="openai", chat_query=THREE_IMAGE_QUERY
        )
        validate_inference_output(model_info, chat_responses, completion_responses, response_snapshot=response_snapshot)


@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, serving_runtime, vllm_inference_service",
    [
        pytest.param(
            {"name": "granite-2b-multi-vision"},
            {"model-dir": MODEL_PATH},
            {"deployment_type": KServeDeploymentType.RAW_DEPLOYMENT},
            {
                "deployment_mode": KServeDeploymentType.RAW_DEPLOYMENT,
                "runtime_argument": SERVING_ARGUMENT,
                "gpu_count": 2,
                "name": "granite-2b-multi",
                "min-replicas": 1,
            },
        ),
    ],
    indirect=True,
)
class TestGraniteMultiGPUVisionModel:
    def test_multi_vision_image_inference(self, vllm_inference_service, get_pod_name_resource, response_snapshot):
        pod = get_pod_name_resource.name
        model_info, chat_responses, completion_responses = run_raw_inference(
            pod_name=pod, isvc=vllm_inference_service, port=8080, endpoint="openai", chat_query=MULTI_IMAGE_QUERIES
        )
        validate_inference_output(model_info, chat_responses, completion_responses, response_snapshot=response_snapshot)
