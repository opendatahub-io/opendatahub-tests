"""vLLM-Omni PVC-backed storage inference validation.

Verifies that the vLLM-Omni runtime correctly serves /v1/audio/speech
requests when model weights are mounted from a PVC (not downloaded from S3).
"""

from typing import Any

import pytest
import requests
import urllib3
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.serving_runtime import ServingRuntime

from tests.model_serving.model_runtime.vllm_omni.constant import QWEN3_TTS_MODEL_PATH
from tests.model_serving.model_runtime.vllm_omni.utils import validate_tts_output
from utilities.constants import KServeDeploymentType
from utilities.inference_utils import get_exposed_isvc_url

urllib3.disable_warnings(category=urllib3.exceptions.InsecureRequestWarning)

QWEN3_TTS_DEFAULT_VOICE: str = "vivian"

PVC_DEPLOYMENT_CONFIG: dict[str, Any] = {
    "deployment_mode": KServeDeploymentType.STANDARD,
    "min-replicas": 1,
}

pytestmark = pytest.mark.usefixtures("skip_if_no_supported_accelerator_type", "valid_aws_config")


@pytest.mark.vllm_omni_nvidia_single_gpu
@pytest.mark.parametrize(
    "model_namespace, vllm_omni_model_pvc, vllm_omni_pvc_downloaded_model_data, "
    "vllm_omni_serving_runtime, vllm_omni_pvc_inference_service",
    [
        pytest.param(
            {"name": "vllm-omni-pvc-qwen3-tts"},
            {"pvc-size": "20Gi"},
            {"model-dir": QWEN3_TTS_MODEL_PATH},
            {"deployment_mode": KServeDeploymentType.STANDARD},
            {
                **PVC_DEPLOYMENT_CONFIG,
                "gpu_count": 1,
                "name": "vllm-omni-pvc-qwen3-tts",
            },
            id="test_vllm_omni_pvc_qwen3_tts_inference",
        ),
    ],
    indirect=True,
)
class TestVllmOmniPvcQwen3TtsInference:
    """Validate vLLM-Omni Qwen3-TTS model inference from PVC-backed storage.

    Steps:
        1. Create a PVC and populate it with Qwen3-TTS model weights downloaded from S3.
        2. Deploy a vLLM-Omni InferenceService pointing to the PVC via pvc:// storage URI.
        3. Send a /v1/audio/speech request over the external route.
        4. Validate the HTTP 200 response carries a valid WAV audio payload.
    """

    def test_vllm_omni_pvc_storage_inference(
        self,
        model_namespace: Namespace,
        vllm_omni_model_pvc: Any,
        vllm_omni_pvc_downloaded_model_data: Any,
        vllm_omni_serving_runtime: ServingRuntime,
        vllm_omni_pvc_inference_service: InferenceService,
    ) -> None:
        """Given a vLLM-Omni ISVC backed by PVC-mounted Qwen3-TTS model weights,
        When a /v1/audio/speech request is sent over the external route,
        Then the service returns HTTP 200 with a valid WAV audio response
        whose body exceeds the 44-byte WAV header minimum.
        """
        base_url: str = get_exposed_isvc_url(isvc=vllm_omni_pvc_inference_service)
        tts_payload: dict[str, str] = {
            "model": vllm_omni_pvc_inference_service.instance.metadata.name,
            "input": "Test",
            "voice": QWEN3_TTS_DEFAULT_VOICE,
            "response_format": "wav",
        }
        response = requests.post(
            url=f"{base_url}/v1/audio/speech",
            json=tts_payload,
            verify=False,
            timeout=60,
        )

        validate_tts_output(response=response)
