import pytest
from simple_logger.logger import get_logger
from utilities.constants import KServeDeploymentType, Ports
from tests.model_serving.model_runtime.vllm.utils import (
    run_raw_inference,
    validate_inference_output,
)
from tests.model_serving.model_runtime.vllm.constant import VLLM_SUPPORTED_QUANTIZATION

LOGGER = get_logger(name=__name__)


SERVING_ARGUMENT = [
    "--model=/mnt/models",
    "--uvicorn-log-level=debug",
    "--chat-template=/opt/app-root/template/tool_chat_template_mistral.jinja",
]

MODEL_PATH = "TheBloke/OpenHermes-2.5-Mistral-7B-AWQ"

pytestmark = pytest.mark.usefixtures("skip_if_no_supported_accelerator_type", "valid_aws_config")


@pytest.mark.vllm_nvidia_single_gpu
@pytest.mark.vllm_amd_gpu
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, serving_runtime, vllm_inference_service",
    [
        pytest.param(
            {"name": "mistral-awq-raw"},
            {"model-dir": MODEL_PATH},
            {"deployment_type": KServeDeploymentType.RAW_DEPLOYMENT},
            {
                "deployment_mode": KServeDeploymentType.RAW_DEPLOYMENT,
                "runtime_argument": SERVING_ARGUMENT,
                "gpu_count": 1,
                "name": "mistralawq-ser-raw",
                "min-replicas": 1,
            },
            id="mistral_raw_deployment",
        ),
        pytest.param(
            {"name": "mistral-marlin-raw"},
            {"model-dir": MODEL_PATH},
            {"deployment_type": KServeDeploymentType.RAW_DEPLOYMENT},
            {
                "deployment_mode": KServeDeploymentType.RAW_DEPLOYMENT,
                "runtime_argument": SERVING_ARGUMENT,
                "gpu_count": 1,
                "quantization": VLLM_SUPPORTED_QUANTIZATION[0],
                "name": "mistralmar-raw",
                "min-replicas": 1,
            },
            id="mistral_raw_marlin_deployment",
        ),
        pytest.param(
            {"name": "mistral-awq-raw"},
            {"model-dir": MODEL_PATH},
            {"deployment_type": KServeDeploymentType.RAW_DEPLOYMENT},
            {
                "deployment_mode": KServeDeploymentType.RAW_DEPLOYMENT,
                "runtime_argument": SERVING_ARGUMENT,
                "gpu_count": 1,
                "quantization": VLLM_SUPPORTED_QUANTIZATION[1],
                "name": "mistralawq-raw",
                "min-replicas": 1,
            },
            id="mistral_raw_awq_deployment",
        ),
    ],
    indirect=True,
)
class TestOpenHermesAWQModel:
    def test_deploy_model_inference_raw(self, vllm_inference_service, vllm_pod_resource, response_snapshot):
        if (
            vllm_inference_service.instance.metadata.annotations["serving.kserve.io/deploymentMode"]
            == KServeDeploymentType.RAW_DEPLOYMENT
        ):
            pod = vllm_pod_resource.name
            model_details, grpc_chat_response, grpc_chat_stream_responses = run_raw_inference(
                pod_name=pod, isvc=vllm_inference_service, port=Ports.GRPC_PORT, endpoint="tgis"
            )
            validate_inference_output(
                model_details, grpc_chat_response, grpc_chat_stream_responses, response_snapshot=response_snapshot
            )

            model_info, chat_responses, completion_responses = run_raw_inference(
                pod_name=pod, isvc=vllm_inference_service, port=Ports.REST_PORT, endpoint="openai"
            )
            validate_inference_output(
                model_info, chat_responses, completion_responses, response_snapshot=response_snapshot
            )
        else:
            pytest.skip("Model deployment is only for kserve raw")


@pytest.mark.vllm_nvidia_multi_gpu
@pytest.mark.vllm_amd_gpu
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, serving_runtime, vllm_inference_service",
    [
        pytest.param(
            {"name": "mistral-marlin-multi"},
            {"model-dir": MODEL_PATH},
            {"deployment_type": KServeDeploymentType.RAW_DEPLOYMENT},
            {
                "deployment_mode": KServeDeploymentType.RAW_DEPLOYMENT,
                "runtime_argument": SERVING_ARGUMENT,
                "gpu_count": 2,
                "quantization": VLLM_SUPPORTED_QUANTIZATION[0],
                "name": "mistralmarlin-raw",
                "min-replicas": 1,
            },
            id="mistral_raw_marlin_multi_gpu_deployment",
        ),
        pytest.param(
            {"name": "mistral-awq-multi"},
            {"model-dir": MODEL_PATH},
            {"deployment_type": KServeDeploymentType.RAW_DEPLOYMENT},
            {
                "deployment_mode": KServeDeploymentType.RAW_DEPLOYMENT,
                "runtime_argument": SERVING_ARGUMENT,
                "gpu_count": 2,
                "quantization": VLLM_SUPPORTED_QUANTIZATION[1],
                "name": "mistralawq-raw",
                "min-replicas": 1,
            },
            id="mistral_raw_awq_multi_gpu_deployment",
        ),
    ],
    indirect=True,
)
class TestOpenHermesAWQMultiGPU:
    def test_deploy_marlin_model_inference_raw(self, vllm_inference_service, vllm_pod_resource, response_snapshot):
        if (
            vllm_inference_service.instance.metadata.annotations["serving.kserve.io/deploymentMode"]
            == KServeDeploymentType.RAW_DEPLOYMENT
        ):
            pod = vllm_pod_resource.name
            model_details, grpc_chat_response, grpc_chat_stream_responses = run_raw_inference(
                pod_name=pod, isvc=vllm_inference_service, port=Ports.GRPC_PORT, endpoint="tgis"
            )

            validate_inference_output(
                model_details, grpc_chat_response, grpc_chat_stream_responses, response_snapshot=response_snapshot
            )
            model_info, chat_responses, completion_responses = run_raw_inference(
                pod_name=pod, isvc=vllm_inference_service, port=Ports.REST_PORT, endpoint="openai"
            )
            validate_inference_output(
                model_info, chat_responses, completion_responses, response_snapshot=response_snapshot
            )
        else:
            pytest.skip("Model deployment is only for kserve raw")
