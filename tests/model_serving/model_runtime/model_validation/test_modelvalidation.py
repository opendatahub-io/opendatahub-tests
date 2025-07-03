from typing import Any
import pytest
from simple_logger.logger import get_logger
from ocp_resources.inference_service import InferenceService
from tests.model_serving.model_runtime.utils import (
    validate_serverless_openai_inference_request,
    validate_raw_openai_inference_request,
)
from tests.model_serving.model_runtime.model_validation.constant import COMPLETION_QUERY
from ocp_resources.pod import Pod

LOGGER = get_logger(name=__name__)


pytestmark = pytest.mark.usefixtures("vllm_skip_if_no_supported_accelerator_type", "valid_aws_config")


@pytest.mark.serverless
class TestVLLMModelcarOCI:
    @pytest.mark.jira("RHOAIENG-25717")
    def test_oci_modelcar_serverless_openai_inference(
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
        )

    def test_oci_modelcar_raw_openai_inference(
        self,
        vllm_model_car_inference_service: InferenceService,
        vllm_pod_resource: Pod,
        response_snapshot: Any,
    ) -> None:
        """
        Validate raw inference request using vLLM runtime and OCI image deployment.
        """
        LOGGER.info("Sending raw inference request to vLLM model served from OCI image.")
        validate_raw_openai_inference_request(
            isvc=vllm_model_car_inference_service,
            pod_name=vllm_pod_resource.name,
            response_snapshot=response_snapshot,
            completion_query=COMPLETION_QUERY,
        )
