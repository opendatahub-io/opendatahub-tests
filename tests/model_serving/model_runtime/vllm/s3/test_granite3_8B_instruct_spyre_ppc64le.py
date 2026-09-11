# noqa: N999
from typing import Any

import pytest
import structlog
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_runtime.vllm.constant import (
    COMPLETION_QUERY,
    GRANITE_CHAT_QUERY,
    GRANITE_SPYRE_RAG_INFERENCE_SERVING_ARGUMENT,
    PREDICT_RESOURCES_RAG_INFERENCE,
)
from tests.model_serving.model_runtime.vllm.utils import validate_raw_openai_inference_request
from utilities.constants import KServeDeploymentType

LOGGER = structlog.get_logger(name=__name__)

MODEL_PATH: str = "models/granite-3.3-8b-instruct"

pytestmark = pytest.mark.usefixtures("skip_if_no_supported_accelerator_type", "valid_aws_config")


@pytest.mark.vllm_spyreppc64le_gpu
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, serving_runtime, vllm_inference_service",
    [
        pytest.param(
            {"name": "granite-spyre-rag"},
            {"model-dir": MODEL_PATH},
            {
                "deployment_mode": KServeDeploymentType.STANDARD,
            },
            {
                "deployment_mode": KServeDeploymentType.STANDARD,
                "runtime_argument": GRANITE_SPYRE_RAG_INFERENCE_SERVING_ARGUMENT,
                "min-replicas": 1,
                "gpu_count": 4,
                "predict_resources": PREDICT_RESOURCES_RAG_INFERENCE,
                "name": "granite-rag-std",
            },
            id="granite-8b-spyre-ppc64le-rag-inference",
        ),
    ],
    indirect=True,
)
class TestGraniteSpyrePpc64leRagInference:
    def test_granite3_8b_spyre_ppc64le_rag_inference(
        self,
        vllm_inference_service: InferenceService,
        skip_if_not_raw_deployment: Any,
        response_snapshot: Any,
    ):
        validate_raw_openai_inference_request(
            isvc=vllm_inference_service,
            response_snapshot=response_snapshot,
            chat_query=GRANITE_CHAT_QUERY,
            completion_query=COMPLETION_QUERY,
        )
