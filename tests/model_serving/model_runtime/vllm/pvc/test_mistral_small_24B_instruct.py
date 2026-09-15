from typing import Any

import pytest
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_runtime.vllm.constant import (
    COMPLETION_QUERY,
    MISTRAL_SPYRE_RAG_INFERENCE_SERVING_ARGUMENT_PVC,
    PREDICT_RESOURCES_RAG_INFERENCE,
)
from tests.model_serving.model_runtime.vllm.utils import validate_raw_openai_inference_request
from utilities.constants import KServeDeploymentType

SERVING_ARGUMENT: list[str] = [
    "--model=/mnt/models",
    "--uvicorn-log-level=debug",
    "--dtype=bfloat16",
]

MISTRAL_CHAT_QUERY: list[list[dict[str, Any]]] = [
    [
        {"role": "user", "content": "What is an even number? Answer in one or two sentences."},
        {"keywords": ["even", "number", "divisible", "two", "2", "integer"]},
    ],
    [
        {"role": "user", "content": "Name three common dog breeds and one trait for each."},
        {"keywords": ["dog", "breed", "retriever", "bulldog", "friendly", "loyal"]},
    ],
]

MODEL_PATH: str = "models/Mistral-Small-3.2-24B-Instruct-2506"


pytestmark = pytest.mark.usefixtures("skip_if_no_supported_accelerator_type", "valid_aws_config")


@pytest.mark.vllm_spyreppc64le_gpu
@pytest.mark.vllm_nvidia_multi_gpu
@pytest.mark.vllm_amd_gpu
@pytest.mark.parametrize(
    "model_namespace, vllm_model_pvc, pvc_downloaded_model_data, serving_runtime, vllm_pvc_inference_service",
    [
        pytest.param(
            {"name": "vllm-pvc-mistral-small-24b"},
            {"pvc-size": "150Gi"},
            {"model-dir": MODEL_PATH},
            {"deployment_mode": KServeDeploymentType.STANDARD},
            {
                "deployment_mode": KServeDeploymentType.STANDARD,
                "runtime_argument": MISTRAL_SPYRE_RAG_INFERENCE_SERVING_ARGUMENT_PVC,
                "min-replicas": 1,
                "gpu_count": 4,
                "predict_resources": PREDICT_RESOURCES_RAG_INFERENCE,
                "name": "vllm-pvc-mistral-small-24b",
                "timeout": 1800,
            },
            id="test_vllm_pvc_mistral_small_24b_spyre_ppc64le",
        ),
    ],
    indirect=True,
)
class TestVllmPvcMistralSmall24BInference:
    """Validate vLLM Mistral-Small 24B model inference from PVC-backed storage.

    Steps:
        1. Create a PVC and download the Mistral-Small-3.2-24B model from S3 into it.
        2. Deploy a vLLM InferenceService using PVC storage with 4 GPUs (TP=4).
        3. Run OpenAI-compatible chat and completion requests.
        4. Validate that inference responses contain expected content.
    """

    def test_vllm_pvc_mistral_small_24b_openai_inference(
        self,
        vllm_pvc_inference_service: InferenceService,
        response_snapshot: Any,
    ) -> None:
        """Given a vLLM ISVC backed by PVC storage with Mistral-Small-3.2-24B,
        When OpenAI-compatible chat and completion requests are sent over the external route,
        Then the model returns valid responses.
        """
        validate_raw_openai_inference_request(
            isvc=vllm_pvc_inference_service,
            response_snapshot=response_snapshot,
            chat_query=MISTRAL_CHAT_QUERY,
            completion_query=COMPLETION_QUERY,
        )
