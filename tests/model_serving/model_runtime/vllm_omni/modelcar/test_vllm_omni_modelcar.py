"""Tests for vLLM-Omni OCI modelcar storage backend."""

from __future__ import annotations

import json
from collections.abc import Generator
from copy import deepcopy
from typing import Any

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime

from tests.model_serving.image_constants import ModelServingImages
from tests.model_serving.model_runtime.vllm.modelcar.constant import (
    MODELCAR_REGISTRIES,
    PULL_SECRET_ACCESS_TYPE,
    PULL_SECRET_NAME,
    SUPPORTED_MODELCAR_REGISTRY_HOSTS,
)
from tests.model_serving.model_runtime.vllm.modelcar.utils import (
    normalize_registry_pull_auth,
    safe_k8s_name,
    validate_registry_pull_auth,
)
from tests.model_serving.model_runtime.vllm.utils import add_image_pull_secrets_if_configured
from tests.model_serving.model_runtime.vllm_omni.constant import (
    AUDIO_SPEECH_ENDPOINT,
    OMNI_SINGLE_GPU_RESOURCES,
    OMNI_VOLUME_MOUNTS,
    OMNI_VOLUMES,
)
from tests.model_serving.model_runtime.vllm_omni.utils import assert_zero_restarts, validate_tts_output
from utilities.constants import KServeDeploymentType, Labels, RuntimeTemplates, Timeout
from utilities.inference_utils import create_isvc, get_exposed_isvc_url
from utilities.serving_runtime import ServingRuntimeFromTemplate

LOGGER = structlog.get_logger(name=__name__)
MODELCAR_ISVC_NAME: str = "vllm-omni-modelcar-qwen3"
TTS_VOICE: str = "vivian"
TTS_INPUT_TEXT: str = "Hello, this is a vLLM-Omni modelcar inference validation."
VALID_AUDIO_CONTENT_TYPES: frozenset[str] = frozenset({"audio/wav", "audio/x-wav"})
pytestmark = pytest.mark.usefixtures("skip_if_no_supported_accelerator_type")


@pytest.fixture(scope="class")
def vllm_omni_modelcar_serving_runtime(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    vllm_omni_runtime_image: str | None,
) -> Generator[ServingRuntime, Any, Any]:
    """vLLM-Omni ServingRuntime deployed from the cluster template."""
    assert model_namespace.name is not None, (
        "model_namespace fixture returned a Namespace with name=None. "
        "Verify the namespace was created successfully and the fixture is configured correctly."
    )
    runtime_kwargs: dict[str, Any] = {
        "client": admin_client,
        "name": "vllm-omni-modelcar-runtime",
        "namespace": model_namespace.name,
        "template_name": RuntimeTemplates.VLLM_OMNI_CUDA,
        "deployment_type": KServeDeploymentType.STANDARD,
    }
    if vllm_omni_runtime_image:
        runtime_kwargs["runtime_image"] = vllm_omni_runtime_image

    with ServingRuntimeFromTemplate(**runtime_kwargs) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def vllm_omni_kserve_registry_pull_secret(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    registry_pull_secret: list[str],
    registry_host: list[str],
) -> Generator[Secret | None, Any, Any]:
    """Dockerconfigjson pull secret for the quay.io/opendatahub OCI registry.

    Yields None when no registry credentials are configured, which causes
    the ISVC fixture to skip pull-secret injection and rely on node-level
    image pull credentials instead.
    """
    if not registry_host:
        yield None
        return

    if len(registry_host) != len(registry_pull_secret):
        raise ValueError(
            f"registry_host count ({len(registry_host)}) must match "
            f"registry_pull_secret count ({len(registry_pull_secret)})"
        )

    unsupported_hosts = set(registry_host) - SUPPORTED_MODELCAR_REGISTRY_HOSTS
    if unsupported_hosts:
        raise ValueError(f"Unsupported OCI registry hosts: {sorted(unsupported_hosts)}")

    auths: dict[str, dict[str, str]] = {}
    for host, raw_auth in zip(registry_host, registry_pull_secret):
        auth = normalize_registry_pull_auth(raw_value=raw_auth, expected_host=host)
        validate_registry_pull_auth(auth=auth)
        auths[host] = {"auth": auth}

    docker_config_json = json.dumps({"auths": auths})
    with Secret(
        client=admin_client,
        name=PULL_SECRET_NAME,
        namespace=model_namespace.name,
        string_data={
            ".dockerconfigjson": docker_config_json,
            "ACCESS_TYPE": PULL_SECRET_ACCESS_TYPE,
            "OCI_HOST": ",".join(registry_host),
        },
        type="kubernetes.io/dockerconfigjson",
        wait_for_resource=True,
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def vllm_omni_modelcar_inference_service(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    vllm_omni_modelcar_serving_runtime: ServingRuntime,
    vllm_omni_kserve_registry_pull_secret: Secret | None,
) -> Generator[InferenceService, Any, Any]:
    """InferenceService backed by the Qwen3-TTS OCI modelcar image.

    Pull secret injection is optional — the default modelcar image is public
    (quay.io/opendatahub). When registry credentials are configured they are
    attached; otherwise the ISVC relies on node-level pull credentials.
    If the image cannot be pulled (ImagePullBackOff), create_isvc will time
    out and the test will fail with a clear deployment error.
    """
    if vllm_omni_kserve_registry_pull_secret is None:
        options_help = ", ".join(f"{registry.cli_option} or {registry.env_var}" for registry in MODELCAR_REGISTRIES)
        LOGGER.warning(
            event="no registry pull secret configured; relying on public access",
            image=ModelServingImages.VLLM_OMNI_MODELCAR,
            options=options_help,
        )

    isvc_kwargs: dict[str, Any] = {
        "client": admin_client,
        "name": safe_k8s_name(MODELCAR_ISVC_NAME),
        "namespace": model_namespace.name,
        "runtime": vllm_omni_modelcar_serving_runtime.name,
        "storage_uri": f"oci://{ModelServingImages.VLLM_OMNI_MODELCAR}",
        "model_format": vllm_omni_modelcar_serving_runtime.instance.spec.supportedModelFormats[0].name,
        "deployment_mode": KServeDeploymentType.STANDARD,
        "external_route": True,
        "resources": deepcopy(x=OMNI_SINGLE_GPU_RESOURCES["resources"]),
        "volumes": OMNI_VOLUMES,
        "volumes_mounts": OMNI_VOLUME_MOUNTS,
        "timeout": Timeout.TIMEOUT_20MIN,
    }
    isvc_kwargs["resources"]["requests"][Labels.Nvidia.NVIDIA_COM_GPU] = 1
    isvc_kwargs["resources"]["limits"][Labels.Nvidia.NVIDIA_COM_GPU] = 1
    add_image_pull_secrets_if_configured(
        isvc_kwargs=isvc_kwargs,
        kserve_registry_pull_secret=vllm_omni_kserve_registry_pull_secret,
    )

    with create_isvc(**isvc_kwargs) as isvc:
        yield isvc


@pytest.mark.vllm_omni_nvidia_single_gpu
@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "vllm-omni-modelcar"},
            id="test_vllm_omni_modelcar_qwen3_tts",
        ),
    ],
    indirect=True,
)
class TestVllmOmniModelcar:
    def test_vllm_omni_modelcar_qwen3_tts_inference(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        vllm_omni_modelcar_inference_service: InferenceService,
    ) -> None:
        """Verify vLLM-Omni inference with OCI modelcar-backed Qwen3-TTS.

        Given a vLLM-Omni InferenceService with storage_uri pointing to an OCI
        modelcar image containing Qwen3-TTS model weights,
        When the InferenceService reaches the Ready state and a /v1/audio/speech
        request is submitted with a TTS prompt,
        Then the response status is HTTP 200, Content-Type is audio/wav or
        audio/x-wav, the response body is non-empty, and the predictor pod
        has zero container restarts.
        """
        LOGGER.info(
            event="sending /v1/audio/speech request to OCI modelcar-backed ISVC",
            image=ModelServingImages.VLLM_OMNI_MODELCAR,
        )

        base_url = get_exposed_isvc_url(isvc=vllm_omni_modelcar_inference_service)
        speech_url = f"{base_url}{AUDIO_SPEECH_ENDPOINT}"
        model_name = vllm_omni_modelcar_inference_service.instance.metadata.name
        payload: dict[str, str] = {
            "model": model_name,
            "input": TTS_INPUT_TEXT,
            "voice": TTS_VOICE,
            "response_format": "wav",
        }

        LOGGER.info(event="POST /v1/audio/speech", url=speech_url, model=model_name, voice=TTS_VOICE)
        response = requests.post(url=speech_url, json=payload, verify=False, timeout=120)
        validate_tts_output(response=response)

        LOGGER.info(
            event="audio response received",
            content_type=response.headers.get("Content-Type", ""),
            size_bytes=len(response.content),
        )

        assert_zero_restarts(client=admin_client, isvc=vllm_omni_modelcar_inference_service)
