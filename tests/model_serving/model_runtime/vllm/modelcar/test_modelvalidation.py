from typing import Any

import pytest
import structlog
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_runtime.utils import (
    validate_raw_openai_inference_request,
)
from tests.model_serving.model_runtime.vllm.modelcar.constant import COMPLETION_QUERY

LOGGER = structlog.get_logger(name=__name__)


pytestmark = pytest.mark.usefixtures("skip_if_no_supported_accelerator_type")


@pytest.mark.vllm_nvidia_single_gpu
@pytest.mark.vllm_amd_gpu
class TestVLLMModelCarRaw:
    def test_oci_model_car_raw_openai_inference(
        self,
        vllm_model_car_inference_service: InferenceService,
        response_snapshot: Any,
        deployment_config: dict[str, Any],
    ) -> None:
        """Given a vLLM ISVC serving a model from an OCI modelcar image with an exposed external route,
        When an OpenAI-compatible completion request is sent over the external route,
        Then the model returns valid inference responses.
        """
        LOGGER.info("Sending inference request to vLLM model served from OCI image via external route.")
        validate_raw_openai_inference_request(
            isvc=vllm_model_car_inference_service,
            model_name=vllm_model_car_inference_service.instance.metadata.name,
            response_snapshot=response_snapshot,
            completion_query=COMPLETION_QUERY,
            model_output_type=deployment_config.get("model_output_type"),
        )
