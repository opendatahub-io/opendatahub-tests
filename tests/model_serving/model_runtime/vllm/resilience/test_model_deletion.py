"""vLLM behavior when underlying model is deleted from S3."""

from typing import Any

import pytest
import requests
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_runtime.vllm.constant import BASE_RAW_DEPLOYMENT_CONFIG
from utilities.constants import Timeout
from utilities.inference_utils import get_exposed_isvc_url

pytestmark = pytest.mark.usefixtures("valid_aws_config")

SIMPLE_CHAT_QUERY: dict[str, Any] = {
    "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
    "max_tokens": 10,
}


@pytest.mark.tier3
@pytest.mark.resilience
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, resilience_serving_runtime, resilience_inference_service",
    [
        pytest.param(
            {"name": "vllm-res-deletion"},
            {"model-dir": "vllm/granite-7b-lab"},
            {**BASE_RAW_DEPLOYMENT_CONFIG},
            {
                "name": "vllm-model-deletion",
                **BASE_RAW_DEPLOYMENT_CONFIG,
                "gpu_count": 1,
                "runtime_argument": [
                    "--model=/mnt/models",
                    "--dtype=float16",
                ],
            },
            id="vllm-model-deletion-resilience",
        ),
    ],
    indirect=True,
)
class TestModelDeletion:
    def test_serving_after_deployment(
        self,
        resilience_inference_service: InferenceService,
    ) -> None:
        """Verify deployed vLLM ISVC responds to inference after initial model download."""
        url = get_exposed_isvc_url(isvc=resilience_inference_service)
        model_name = resilience_inference_service.name
        query = {**SIMPLE_CHAT_QUERY, "model": model_name}

        response = requests.post(f"{url}/v1/chat/completions", json=query, verify=False, timeout=Timeout.TIMEOUT_2MIN)
        assert response.status_code == 200, (
            f"Expected 200 for baseline inference, got {response.status_code}: {response.text[:200]}"
        )
