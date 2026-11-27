"""vLLM-Omni fast template inference tests.

Validates that the fast-1 and fast-2 vLLM-Omni ServingRuntime templates
can deploy Qwen3-TTS-1.7B and serve a basic TTS request.

These tests are deselected by pytest_collection_modifyitems in the parent
conftest when the fast template does not exist on the cluster.
"""

import pytest
import requests
import urllib3
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_runtime.vllm_omni.constant import (
    AUDIO_SPEECH_ENDPOINT,
    QWEN3_TTS_MODEL_PATH,
)
from tests.model_serving.model_runtime.vllm_omni.utils import assert_health_ok, validate_tts_output
from utilities.constants import KServeDeploymentType, RuntimeTemplates

urllib3.disable_warnings(category=urllib3.exceptions.InsecureRequestWarning)

VERSION_ENDPOINT: str = "/version"

pytestmark = pytest.mark.usefixtures("skip_if_no_supported_accelerator_type", "valid_aws_config")


@pytest.mark.tier1
@pytest.mark.vllm_omni_nvidia_single_gpu
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, vllm_omni_serving_runtime, vllm_omni_inference_service",
    [
        pytest.param(
            {"name": "vllm-omni-fast-1-inference"},
            {"model-dir": QWEN3_TTS_MODEL_PATH},
            {
                "template_name": RuntimeTemplates.VLLM_OMNI_CUDA_FAST_1,
                "deployment_type": KServeDeploymentType.STANDARD,
            },
            {
                "name": "vllm-omni-fast-1-isvc",
                "model_path": QWEN3_TTS_MODEL_PATH,
                "deployment_mode": KServeDeploymentType.STANDARD,
                "gpu_count": 1,
            },
            id="test_vllm_omni_fast_1_cuda_inference",
        ),
        pytest.param(
            {"name": "vllm-omni-fast-2-inference"},
            {"model-dir": QWEN3_TTS_MODEL_PATH},
            {
                "template_name": RuntimeTemplates.VLLM_OMNI_CUDA_FAST_2,
                "deployment_type": KServeDeploymentType.STANDARD,
            },
            {
                "name": "vllm-omni-fast-2-isvc",
                "model_path": QWEN3_TTS_MODEL_PATH,
                "deployment_mode": KServeDeploymentType.STANDARD,
                "gpu_count": 1,
            },
            id="test_vllm_omni_fast_2_cuda_inference",
        ),
    ],
    indirect=True,
)
class TestVllmOmniFastInference:
    """Validate vLLM-Omni inference using fast ServingRuntime templates."""

    def test_fast_health(self, vllm_omni_inference_service: InferenceService, vllm_omni_isvc_url: str) -> None:
        """Given a vLLM-Omni ISVC deployed with a fast runtime template,
        When a GET /health request is sent,
        Then the response is HTTP 200.
        """
        assert_health_ok(url=vllm_omni_isvc_url)

    def test_fast_tts_inference(
        self,
        vllm_omni_inference_service: InferenceService,
        vllm_omni_isvc_url: str,
    ) -> None:
        """Given a vLLM-Omni ISVC deployed with a fast runtime template,
        When a TTS request is sent to /v1/audio/speech,
        Then the response is HTTP 200 with valid audio content.
        """
        payload = {
            "model": vllm_omni_inference_service.instance.metadata.name,
            "input": "Fast template inference test.",
            "voice": "vivian",
        }
        response = requests.post(
            f"{vllm_omni_isvc_url}{AUDIO_SPEECH_ENDPOINT}",
            json=payload,
            verify=False,
            timeout=120,
        )
        validate_tts_output(response=response)

    def test_fast_vllm_omni_version(
        self, vllm_omni_inference_service: InferenceService, vllm_omni_isvc_url: str
    ) -> None:
        """Verify the fast template serves a valid vLLM-Omni version."""
        response = requests.get(f"{vllm_omni_isvc_url}{VERSION_ENDPOINT}", verify=False, timeout=30)
        response.raise_for_status()
        version_data = response.json()
        assert "version" in version_data, f"No 'version' key in {VERSION_ENDPOINT} response: {version_data}"
        assert version_data["version"], f"vLLM-Omni version is empty: {version_data}"
