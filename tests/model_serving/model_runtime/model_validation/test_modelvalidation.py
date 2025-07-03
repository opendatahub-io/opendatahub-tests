import pytest
from simple_logger.logger import get_logger
from typing import Any
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from utilities.constants import KServeDeploymentType, Protocols
from tests.model_serving.model_runtime.model_validation.utils import validate_inference_request
from tests.model_serving.model_runtime.model_validation.constant import (
    COMPLETION_QUERY,
    BASE_SEVERRLESS_DEPLOYMENT_CONFIG,
)


LOGGER = get_logger(name=__name__)

# TIMEOUT_20MIN: str = 30 * 60

SERVING_ARGUMENT: list[str] = [
    "--uvicorn-log-level=debug",
    "--tensor-parallel-size=2",
    "--max-model-len=1024",
    "--trust-remote-code",
    "--distributed-executor-backend=mp",
]

BASE_SEVERRLESS_DEPLOYMENT_CONFIG["runtime_argument"] = SERVING_ARGUMENT

pytestmark = pytest.mark.usefixtures("skip_if_no_supported_accelerator_type", "valid_registry_pullsecret")

OCI_IMAGE_NAME = "registry.redhat.io/rhelai1/modelcar-granite-3-1-8b-instruct-quantized-w4a16:1.5"


@pytest.mark.serverless
@pytest.mark.parametrize(
    "vllm_serverless_inference_service",
    [
        pytest.param(
            {"name": "granite-3.1-8b-instruct-quantized-w4a16"},
            {"model-dir": OCI_IMAGE_NAME},
            {"deployment_type": KServeDeploymentType.SERVERLESS},
            {
                **BASE_SEVERRLESS_DEPLOYMENT_CONFIG,
                "gpu_count": 2,
                "name": "granite-3.1-8b-instruct-quantized-w4a16",
            },
            id="granite-3.1-8b-instruct-quantized-w4a16",
        )
    ],
    indirect=True,
)
class ModelcarValidation:
    def test_modelcar_image(
        self,
        vllm_serverless_inference_service: InferenceService,
        vllm_pod_resource: Pod,
        response_snapshot: Any,
    ) -> None:
        # Create input_data with the completion query
        input_data = {
            "prompt": COMPLETION_QUERY[0]["text"],  # Use the first prompt from COMPLETION_QUERY
            "max_tokens": 256,
        }

        validate_inference_request(
            pod_name=vllm_pod_resource.name,  # Get pod name from the pod resource
            isvc=vllm_serverless_inference_service,
            response_snapshot=response_snapshot,
            input_query=input_data,
            protocol=Protocols.HTTPS,
        )
