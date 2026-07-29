"""E2e smoke test for KV cache filesystem secondary tier on a GPU cluster."""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.llm_inference_service import LLMInferenceService

from tests.model_serving.model_server.llmd.llmd_configs import TinyLlamaOciGpuConfig
from tests.model_serving.model_server.llmd.utils import (
    get_llmd_inference_pool_pods,
    ns_from_file,
    parse_completion_text,
    send_chat_completions,
    workaround_503_no_healthy_upstream,
)

pytestmark = [pytest.mark.llmd_gpu]

NAMESPACE = ns_from_file(file=__file__)

_VOLUME_NAME = "kv-cache-secondary-0"
_MOUNT_PATH = "/mnt/kv-cache-0"
_CONTAINER_NAME = "main"

_KvCacheDiskOffloadConfig = TinyLlamaOciGpuConfig.with_overrides(
    name="llmisvc-kv-cache-disk-offload",
    kv_cache_offloading={
        "cpu": "4Gi",
        "secondary": [
            {"fileSystem": {"emptyDir": {"size": "20Gi"}}},
        ],
    },
)


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llmisvc",
    [({"name": NAMESPACE}, _KvCacheDiskOffloadConfig)],
    indirect=True,
)
@pytest.mark.usefixtures("skip_if_disconnected")
class TestLlmdSinglenodeKvCacheDiskOffload:
    """Deploy TinyLlama on GPU with a filesystem secondary KV cache tier.

    Validates that the controller attaches an emptyDir-backed ephemeral volume,
    requests matching ephemeral-storage on the container, and that vLLM starts
    and serves requests successfully.
    """

    def test_kv_cache_disk_volume_attached(
        self,
        unprivileged_client: DynamicClient,
        llmisvc: LLMInferenceService,
    ):
        """Verify every workload pod has the ephemeral secondary KV cache volume and mount."""
        pods = get_llmd_inference_pool_pods(client=unprivileged_client, llmisvc=llmisvc)
        assert pods, f"No workload pods found for {llmisvc.name}"

        for pod in pods:
            spec = pod.instance.spec

            volume_names = [v.name for v in (spec.volumes or [])]
            assert _VOLUME_NAME in volume_names, (
                f"Pod {pod.name}: expected ephemeral volume '{_VOLUME_NAME}'; got {volume_names}"
            )

            ephemeral_vol = next(v for v in spec.volumes if v.name == _VOLUME_NAME)
            assert (
                getattr(ephemeral_vol, "emptyDir", None) is not None
                or getattr(ephemeral_vol, "ephemeral", None) is not None
            ), f"Pod {pod.name}: volume '{_VOLUME_NAME}' is not a local storage volume (got {dict(ephemeral_vol)})"

            container = next(
                (c for c in (spec.containers or []) if c.name == _CONTAINER_NAME),
                None,
            )
            assert container is not None, f"Pod {pod.name}: container '{_CONTAINER_NAME}' not found"

            mount_paths = [m.mountPath for m in (container.volumeMounts or [])]
            assert _MOUNT_PATH in mount_paths, f"Pod {pod.name}: expected mount at '{_MOUNT_PATH}'; got {mount_paths}"

            requests = dict(container.resources.requests or {}) if container.resources else {}
            assert "ephemeral-storage" in requests, (
                f"Pod {pod.name}: expected ephemeral-storage resource request; got {requests}"
            )

    def test_llmd_singlenode_kv_cache_disk_offload(
        self,
        llmisvc: LLMInferenceService,
    ):
        """Verify inference succeeds with the secondary disk KV cache tier active."""
        prompt = "What is the capital of Italy?"
        expected = "rome"

        workaround_503_no_healthy_upstream(llmisvc=llmisvc, prompt=prompt)

        status, body = send_chat_completions(llmisvc=llmisvc, prompt=prompt)
        assert status == 200, f"Expected 200, got {status}: {body}"
        completion = parse_completion_text(response_body=body)
        assert expected in completion.lower(), f"Expected '{expected}' in response, got: {completion}"
