from typing import Any
import pytest
from simple_logger.logger import get_logger
from ocp_resources.inference_service import InferenceService
from utilities.constants import KServeDeploymentType
from tests.model_serving.model_runtime.model_validation.utils import validate_serverless_openai_inference_request
from tests.model_serving.model_runtime.model_validation.constant import COMPLETION_QUERY

LOGGER = get_logger(name=__name__)

TIMEOUT_20MIN: str = 30 * 60

# SERVING_ARGUMENT: list[str] = [
#     "--uvicorn-log-level=debug",
#     "--max-model-len=1024",
#     "--trust-remote-code",
#     "--distributed-executor-backend=mp",
# ]

# BASE_SEVERRLESS_DEPLOYMENT_CONFIG["runtime_argument"] = SERVING_ARGUMENT

pytestmark = pytest.mark.usefixtures(
    "vllm_skip_if_no_supported_accelerator_type", "valid_aws_config", "valid_registry_pullsecret"
)


@pytest.mark.serverless
@pytest.mark.parametrize(
    "dynamic_model_namespace, modelcar_serving_runtime, vllm_model_car_inference_service, modelcar_image_uri",
    # user can override the modelcar_image_uri in the test case from the cli input
    # Maximum of 10 modelcar images can be tested at once
    [
        pytest.param(
            {"modelcar_image_uri": "modelcar-granite-3-1-8b-base-quantized-w4a16:1.5", "modelmesh-enabled": False},
            {"deployment_type": KServeDeploymentType.SERVERLESS},
            {
                "modelcar_image_uri": "modelcar-granite-3-1-8b-base-quantized-w4a16:1.5",
                "gpu_count": 1,
            },
            "modelcar-granite-3-1-8b-base-quantized-w4a16:1.5",
            id="granite-8b-oci",
        ),
        pytest.param(
            {"modelcar_image_uri": "modelcar-llama-3-1-8b-instruct:1.5", "modelmesh-enabled": False},
            {"deployment_type": KServeDeploymentType.SERVERLESS},
            {
                "modelcar_image_uri": "modelcar-llama-3-1-8b-instruct:1.5",
                "gpu_count": 1,
            },
            "modelcar-llama-3-1-8b-instruct:1.5",
            id="llama-8b-oci",
        ),
        pytest.param(
            {"modelcar_image_uri": "modelcar-qwen2-5-7b-instruct:1.5", "modelmesh-enabled": False},
            {"deployment_type": KServeDeploymentType.SERVERLESS},
            {
                "modelcar_image_uri": "modelcar-qwen2-5-7b-instruct:1.5",
                "gpu_count": 1,
                "timeout": TIMEOUT_20MIN,
            },
            "modelcar-qwen2-5-7b-instruct:1.5",
            id="qwen2-7b-oci",
        ),
        pytest.param(
            {"modelcar_image_uri": "modelcar-mistral-7b-instruct:1.5", "modelmesh-enabled": False},
            {"deployment_type": KServeDeploymentType.SERVERLESS},
            {
                "modelcar_image_uri": "modelcar-mistral-7b-instruct:1.5",
                "gpu_count": 1,
                "timeout": TIMEOUT_20MIN,
            },
            "modelcar-mistral-7b-instruct:1.5",
            id="mistral-7b-oci",
        ),
        pytest.param(
            {"modelcar_image_uri": "modelcar-mistral-7b-instruct:1.5", "modelmesh-enabled": False},
            {"deployment_type": KServeDeploymentType.SERVERLESS},
            {
                "modelcar_image_uri": "modelcar-mistral-7b-instruct:1.5",
                "gpu_count": 1,
                "timeout": TIMEOUT_20MIN,
            },
            "modelcar-mistral-7b-instruct:1.5",
            id="mistral-7b-oci",
        ),
        pytest.param(
            {"modelcar_image_uri": "modelcar-llama-3-1-8b-instruct:1.5", "modelmesh-enabled": False},
            {"deployment_type": KServeDeploymentType.SERVERLESS},
            {
                "modelcar_image_uri": "modelcar-llama-3-1-8b-instruct:1.5",
                "gpu_count": 1,
            },
            "modelcar-llama-3-1-8b-instruct:1.5",
            id="llama-8b-oci",
        ),
        pytest.param(
            {"modelcar_image_uri": "modelcar-llama-3-2-70b-instruct:1.5", "modelmesh-enabled": False},
            {"deployment_type": KServeDeploymentType.SERVERLESS},
            {
                "modelcar_image_uri": "modelcar-llama-3-2-70b-instruct:1.5",
                "gpu_count": 1,
                "timeout": TIMEOUT_20MIN,
            },
            "modelcar-llama-3-2-70b-instruct:1.5",
            id="llama-70b-oci",
        ),
        pytest.param(
            {"modelcar_image_uri": "modelcar-llama-3-2-70b-instruct:1.5", "modelmesh-enabled": False},
            {"deployment_type": KServeDeploymentType.SERVERLESS},
            {
                "modelcar_image_uri": "modelcar-llama-3-2-70b-instruct:1.5",
                "gpu_count": 1,
                "timeout": TIMEOUT_20MIN,
            },
            "modelcar-llama-3-2-70b-instruct:1.5",
            id="llama-70b-oci",
        ),
        pytest.param(
            {"modelcar_image_uri": "modelcar-llama-3-2-70b-instruct:1.5", "modelmesh-enabled": False},
            {"deployment_type": KServeDeploymentType.SERVERLESS},
            {
                "modelcar_image_uri": "modelcar-llama-3-2-70b-instruct:1.5",
                "gpu_count": 1,
                "timeout": TIMEOUT_20MIN,
            },
            "modelcar-llama-3-2-70b-instruct:1.5",
            id="llama-70b-oci",
        ),
        pytest.param(
            {"modelcar_image_uri": "modelcar-llama-3-2-70b-instruct:1.5", "modelmesh-enabled": False},
            {"deployment_type": KServeDeploymentType.SERVERLESS},
            {
                "modelcar_image_uri": "modelcar-llama-3-2-70b-instruct:1.5",
                "gpu_count": 1,
                "timeout": TIMEOUT_20MIN,
            },
            "modelcar-llama-3-2-70b-instruct:1.5",
            id="llama-70b-oci",
        ),
    ],
    indirect=True,
)
class TestVLLMModelcarOCI:
    @pytest.mark.jira("RHOAIENG-25717")
    def test_openai_completion_from_oci_image(
        self,
        vllm_model_car_inference_service: InferenceService,
        response_snapshot: Any,
    ) -> None:
        """
        Validate OpenAI-style completion request using vLLM runtime and OCI image deployment.
        """
        LOGGER.info("Sending OpenAI-style completion request to vLLM model served from OCI image.")
        validate_serverless_openai_inference_request(
            url=vllm_model_car_inference_service.instance.status.url,
            model_name=vllm_model_car_inference_service.instance.metadata.name,
            response_snapshot=response_snapshot,
            completion_query=COMPLETION_QUERY,
            chat_query=None,
        )
