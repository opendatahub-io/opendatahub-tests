from typing import Any

import pytest
import requests
import structlog
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_runtime.vllm.constant import CHAT_QUERY, COMPLETION_QUERY
from tests.model_serving.model_runtime.vllm.utils import validate_raw_openai_inference_request
from utilities.constants import KServeDeploymentType, RuntimeTemplates
from utilities.inference_utils import get_exposed_isvc_url

LOGGER = structlog.get_logger(name=__name__)

MODEL_PATH: str = "granite-7b-starter"

FAST_SERVING_ARGUMENT: list[str] = [
    "--model=/mnt/models",
    "--dtype=float16",
]

FAST_RAW_DEPLOYMENT_CONFIG: dict[str, Any] = {
    "deployment_mode": KServeDeploymentType.STANDARD,
    "runtime_argument": FAST_SERVING_ARGUMENT,
    "min-replicas": 1,
}

pytestmark = pytest.mark.usefixtures("skip_if_no_supported_accelerator_type", "valid_aws_config")


def _assert_vllm_version_reported(isvc: InferenceService) -> None:
    """Query the vLLM /version endpoint and verify it returns a non-empty version."""
    url = get_exposed_isvc_url(isvc=isvc)
    response = requests.get(f"{url}/version", verify=False, timeout=30)
    response.raise_for_status()
    version_info = response.json()
    assert "version" in version_info, f"Expected 'version' key in response, got: {version_info}"
    version = version_info["version"]
    assert version, f"vLLM version is empty: {version_info}"
    LOGGER.info("Reported vLLM version: %s", version)


@pytest.mark.tier1
@pytest.mark.vllm_nvidia_single_gpu
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, serving_runtime, vllm_inference_service",
    [
        pytest.param(
            {"name": "vllm-fast-1-inference"},
            {"model-dir": MODEL_PATH},
            {
                "template_name": RuntimeTemplates.VLLM_FAST_1_CUDA,
                "deployment_type": KServeDeploymentType.STANDARD,
            },
            {
                **FAST_RAW_DEPLOYMENT_CONFIG,
                "gpu_count": 1,
                "name": "vllm-fast-1-isvc",
            },
            id="test_vllm_fast_1_cuda_inference",
        ),
    ],
    indirect=True,
)
class TestVllmFast1Inference:
    """Validate vLLM inference using the fast-1 ServingRuntime template."""

    def test_fast_1_openai_inference(
        self,
        vllm_inference_service: InferenceService,
        response_snapshot: Any,
    ) -> None:
        """Given a vLLM ISVC deployed with the fast-1 runtime template,
        When OpenAI-compatible chat and completion requests are sent,
        Then the model returns valid responses.
        """
        validate_raw_openai_inference_request(
            isvc=vllm_inference_service,
            response_snapshot=response_snapshot,
            chat_query=CHAT_QUERY,
            completion_query=COMPLETION_QUERY,
        )

    def test_fast_1_vllm_version(
        self,
        vllm_inference_service: InferenceService,
    ) -> None:
        """Given a vLLM ISVC deployed with the fast-1 runtime template,
        When the /version endpoint is queried,
        Then a valid vLLM version string is returned.
        """
        _assert_vllm_version_reported(isvc=vllm_inference_service)


@pytest.mark.tier1
@pytest.mark.vllm_nvidia_single_gpu
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, serving_runtime, vllm_inference_service",
    [
        pytest.param(
            {"name": "vllm-fast-2-inference"},
            {"model-dir": MODEL_PATH},
            {
                "template_name": RuntimeTemplates.VLLM_FAST_2_CUDA,
                "deployment_type": KServeDeploymentType.STANDARD,
            },
            {
                **FAST_RAW_DEPLOYMENT_CONFIG,
                "gpu_count": 1,
                "name": "vllm-fast-2-isvc",
            },
            id="test_vllm_fast_2_cuda_inference",
        ),
    ],
    indirect=True,
)
class TestVllmFast2Inference:
    """Validate vLLM inference using the fast-2 ServingRuntime template."""

    def test_fast_2_openai_inference(
        self,
        vllm_inference_service: InferenceService,
        response_snapshot: Any,
    ) -> None:
        """Given a vLLM ISVC deployed with the fast-2 runtime template,
        When OpenAI-compatible chat and completion requests are sent,
        Then the model returns valid responses.
        """
        validate_raw_openai_inference_request(
            isvc=vllm_inference_service,
            response_snapshot=response_snapshot,
            chat_query=CHAT_QUERY,
            completion_query=COMPLETION_QUERY,
        )

    def test_fast_2_vllm_version(
        self,
        vllm_inference_service: InferenceService,
    ) -> None:
        """Given a vLLM ISVC deployed with the fast-2 runtime template,
        When the /version endpoint is queried,
        Then a valid vLLM version string is returned.
        """
        _assert_vllm_version_reported(isvc=vllm_inference_service)
