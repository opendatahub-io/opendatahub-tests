"""FLUX.2 image generation tests via the vLLM-Omni runtime."""

import base64
from typing import Any

import pytest
import requests
import structlog
import urllib3
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from urllib3.exceptions import InsecureRequestWarning

from tests.model_serving.model_runtime.utils import get_restart_counts
from tests.model_serving.model_runtime.vllm_omni.constant import FLUX2_MODEL_PATH, IMAGES_GENERATIONS_ENDPOINT
from tests.model_serving.model_runtime.vllm_omni.utils import _JPEG_MAGIC, _PNG_MAGIC, assert_no_pod_restarts
from utilities.constants import KServeDeploymentType

urllib3.disable_warnings(category=InsecureRequestWarning)

LOGGER = structlog.get_logger(name=__name__)

pytestmark = pytest.mark.usefixtures("skip_if_no_supported_accelerator_type", "valid_aws_config")


@pytest.mark.vllm_omni_nvidia_single_gpu
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, vllm_omni_serving_runtime, vllm_omni_inference_service",
    [
        pytest.param(
            {"name": "vllm-omni-flux2-img"},
            {"model-dir": FLUX2_MODEL_PATH},
            {"deployment_mode": KServeDeploymentType.STANDARD},
            {
                "deployment_mode": KServeDeploymentType.STANDARD,
                "name": "vllm-omni-flux2",
                "min-replicas": 1,
                "model_path": FLUX2_MODEL_PATH,
                "gpu_count": 1,
            },
            id="test_vllm_omni_flux2_image_generation",
        ),
    ],
    indirect=True,
)
class TestVllmOmniFLUX2ImageGeneration:
    """Validate FLUX.2 image generation via vLLM-Omni runtime.

    Deploys FLUX.2-klein-4B on 1-GPU hardware and verifies that the
    /v1/images/generations endpoint returns a valid base64-encoded PNG or JPEG
    image without making any quality or accuracy assertions.
    """

    def test_vllm_omni_flux2_image_generation(
        self,
        vllm_omni_inference_service: InferenceService,
        vllm_omni_isvc_url: str,
        vllm_omni_pod_resource: Pod,
    ) -> None:
        """Given a vLLM-Omni InferenceService backed by the FLUX.2-klein-4B model,
        When a POST /v1/images/generations request is sent with a simple prompt,
        Then the response is HTTP 200 with a data array containing a valid b64_json
        PNG or JPEG image, and no pod restarts occur.
        """
        initial_restarts = get_restart_counts(pod=vllm_omni_pod_resource)
        base_url: str = vllm_omni_isvc_url
        url: str = f"{base_url}{IMAGES_GENERATIONS_ENDPOINT}"
        payload: dict[str, Any] = {
            "model": vllm_omni_inference_service.instance.metadata.name,
            "prompt": "A simple red circle on white background",
            "size": "512x512",
            "seed": 42,
            "n": 1,
        }

        response: requests.Response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            verify=False,
            timeout=120,
        )
        LOGGER.info(event="FLUX.2 image generation response", status_code=response.status_code)

        assert response.status_code == 200, (
            f"Expected HTTP 200 from {IMAGES_GENERATIONS_ENDPOINT}, got {response.status_code}; "
            f"body: {response.text[:500]}"
        )
        assert "application/json" in response.headers.get("Content-Type", ""), (
            f"Expected Content-Type application/json, got: {response.headers.get('Content-Type')}"
        )

        body: dict[str, Any] = response.json()
        data: list[dict[str, Any]] = body.get("data", [])
        assert len(data) >= 1, f"Expected at least 1 element in 'data', got: {data}"

        b64_json: str = data[0].get("b64_json", "")
        assert b64_json, (
            f"data[0].b64_json is empty or missing in {IMAGES_GENERATIONS_ENDPOINT} response. "
            f"Got data[0] keys: {list(data[0].keys())}. "
            "Verify the FLUX.2 model supports b64_json response format."
        )

        raw_bytes: bytes = base64.b64decode(b64_json)
        is_png: bool = raw_bytes[:4] == _PNG_MAGIC
        is_jpeg: bool = raw_bytes[:3] == _JPEG_MAGIC
        assert is_png or is_jpeg, (
            f"Decoded b64_json does not start with PNG ({_PNG_MAGIC.hex()}) or "
            f"JPEG ({_JPEG_MAGIC.hex()}) magic bytes; "
            f"first 8 bytes: {raw_bytes[:8].hex()}"
        )

        assert_no_pod_restarts(
            pod=vllm_omni_pod_resource,
            initial_counts=initial_restarts,
            context="FLUX.2 image generation",
        )
