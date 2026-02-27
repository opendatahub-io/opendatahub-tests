import pytest
from simple_logger.logger import get_logger
from typing import List, Any, Generator
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from utilities.constants import KServeDeploymentType, Ports
from tests.model_serving.model_runtime.vllm.utils import (
    run_raw_inference,
    validate_inference_output,
)
from tests.model_serving.model_runtime.vllm.constant import OPENAI_ENDPOINT_NAME, TGIS_ENDPOINT_NAME

LOGGER = get_logger(name=__name__)

TIMEOUT_20MIN: str = 20 * 60
SERVING_ARGUMENT: List[str] = [
    "--model=/mnt/models/granite-7b-instruct",
    "--uvicorn-log-level=debug",
    "--dtype=float16",
    "--speculative_config",
    '{ "model": "/mnt/models/granite-7b-instruct-accelerator", "num_speculative_tokens": 5 }',
]

MODEL_PATH: str = "speculative_decoding"


pytestmark = pytest.mark.usefixtures("skip_if_no_supported_accelerator_type", "valid_aws_config")


@pytest.mark.vllm_nvidia_single_gpu
@pytest.mark.vllm_amd_gpu
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, serving_runtime, vllm_inference_service",
    [
        pytest.param(
            {"name": "granite-7b-lab-draft"},
            {"model-dir": MODEL_PATH},
            {"deployment_type": KServeDeploymentType.RAW_DEPLOYMENT},
            {
                "deployment_mode": KServeDeploymentType.RAW_DEPLOYMENT,
                "runtime_argument": SERVING_ARGUMENT,
                "gpu_count": 1,
                "name": "granite-7b",
                "min-replicas": 1,
                "timeout": TIMEOUT_20MIN,
            },
        ),
    ],
    indirect=True,
)
class TestGraniteLabDraftModel:
    @pytest.mark.xfail(
        reason="vLLM does not support MLPSpeculatorPreTrainedModel architecture for draft model speculative decoding"
    )
    def test_spec_draft_inference(
        self,
        vllm_inference_service: Generator[InferenceService, Any, Any],
        vllm_pod_resource: Pod,
        response_snapshot: Any,
    ):
        model_info, chat_responses, completion_responses = run_raw_inference(
            pod_name=vllm_pod_resource.name,
            isvc=vllm_inference_service,
            port=Ports.REST_PORT,
            endpoint=OPENAI_ENDPOINT_NAME,
        )
        model_info_tgis, completion_responses_tgis, completion_responses_tgis_stream = run_raw_inference(
            pod_name=vllm_pod_resource.name,
            isvc=vllm_inference_service,
            port=Ports.GRPC_PORT,
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


@pytest.mark.vllm_nvidia_multi_gpu
@pytest.mark.vllm_amd_gpu
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, serving_runtime, vllm_inference_service",
    [
        pytest.param(
            {"name": "granite-7b-lab-m-draft"},
            {"model-dir": MODEL_PATH},
            {"deployment_type": KServeDeploymentType.RAW_DEPLOYMENT},
            {
                "deployment_mode": KServeDeploymentType.RAW_DEPLOYMENT,
                "runtime_argument": SERVING_ARGUMENT,
                "gpu_count": 2,
                "name": "granite-7b",
                "min-replicas": 1,
                "timeout": TIMEOUT_20MIN,
            },
        ),
    ],
    indirect=True,
)
class TestMultiGraniteLabDraftModel:
    @pytest.mark.xfail(
        reason="vLLM does not support MLPSpeculatorPreTrainedModel architecture for draft model speculative decoding"
    )
    def test_multi_spec_draft_inference(
        self,
        vllm_inference_service: Generator[InferenceService, Any, Any],
        vllm_pod_resource: Pod,
        response_snapshot: Any,
    ):
        model_info, chat_responses, completion_responses = run_raw_inference(
            pod_name=vllm_pod_resource.name,
            isvc=vllm_inference_service,
            port=Ports.REST_PORT,
            endpoint=OPENAI_ENDPOINT_NAME,
        )
        model_info_tgis, completion_responses_tgis, completion_responses_tgis_stream = run_raw_inference(
            pod_name=vllm_pod_resource.name,
            isvc=vllm_inference_service,
            port=Ports.GRPC_PORT,
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
