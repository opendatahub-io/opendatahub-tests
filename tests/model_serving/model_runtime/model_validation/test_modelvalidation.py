from typing import Any

import pytest
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from simple_logger.logger import get_logger

from tests.model_serving.model_runtime.model_validation.constant import COMPLETION_QUERY
from tests.model_serving.model_runtime.utils import (
    validate_raw_openai_inference_request,
)

LOGGER = get_logger(name=__name__)


pytestmark = pytest.mark.usefixtures("skip_if_no_supported_accelerator_type")


@pytest.mark.model_validation
class TestVLLMModelCarRaw:
    def test_oci_model_car_raw_openai_inference(
        self,
        vllm_model_car_inference_service: InferenceService,
        vllm_model_car_pod_resource: Pod,
        response_snapshot: Any,
        deployment_config: dict[str, Any],
    ) -> None:
        """
        Validate raw inference request using vLLM runtime and OCI image deployment.
        """
        LOGGER.info("Sending raw inference request to vLLM model served from OCI image.")
        validate_raw_openai_inference_request(
            isvc=vllm_model_car_inference_service,
            pod_name=vllm_model_car_pod_resource.name,
            model_name=vllm_model_car_inference_service.instance.metadata.name,
            response_snapshot=response_snapshot,
            completion_query=COMPLETION_QUERY,
            model_output_type=deployment_config.get("model_output_type"),
        )
