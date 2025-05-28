import pytest
from simple_logger.logger import get_logger
from typing import Any, Generator
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from utilities.constants import KServeDeploymentType
from tests.model_serving.model_runtime.vllm.utils import (
    validate_raw_openai_inference_request 
)
from tests.model_serving.model_runtime.vllm.constant import (
    COMPLETION_QUERY,
    BASE_SEVERRLESS_DEPLOYMENT_CONFIG,
)

LOGGER = get_logger(name=__name__)

TIMEOUT_20MIN: str = 30 * 60

SERVING_ARGUMENT: list[str] = [
    "--uvicorn-log-level=debug",
    "--tensor-parallel-size=2",
    "--max-model-len=1024",
    "--trust-remote-code",
    "--distributed-executor-backend=mp"
]

BASE_SEVERRLESS_DEPLOYMENT_CONFIG["runtime_argument"] = SERVING_ARGUMENT

pytestmark = pytest.mark.usefixtures("skip_if_no_supported_accelerator_type")

OCI_IMAGE_NAME ="registry.redhat.io/rhelai1/modelcar-granite-3-1-8b-instruct-quantized-w4a16:1.5"

@pytest.mark.serverless
@pytest.mark.parametrize(
    "vllm_serverless_inference_service", [
        pytest.param(
            {
                "deployment_config": BASE_SEVERRLESS_DEPLOYMENT_CONFIG,
                "model_config": {
                    "storage_uri": f"oci://{OCI_IMAGE_NAME}",
                },
            },
            id="granite-8b-oci"
        )
    ],
    indirect=True,
)

class TestVLLMModelcarOCI:
    @pytest.mark.jira("RHOAIENG-25717")  
    def test_openai_completion_from_oci_image(self, vllm_serverless_inference_service: InferenceService) -> None:
        """
        Validate OpenAI-style completion request using vLLM runtime and OCI image deployment.
        """
        LOGGER.info("Sending OpenAI-style completion request to vLLM model served from OCI image.")
        validate_raw_openai_inference_request(
            inference_service=vllm_serverless_inference_service,
            query=COMPLETION_QUERY,
        )