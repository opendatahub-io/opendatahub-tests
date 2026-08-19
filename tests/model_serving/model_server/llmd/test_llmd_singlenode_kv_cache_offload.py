"""E2e smoke tests for KV cache offloading (CPU and disk tiers) on a GPU cluster."""

import pytest
from kubernetes.dynamic import DynamicClient

from tests.model_serving.model_server.llmd.llmd_configs import KvCacheCpuOffloadConfig, KvCacheDiskOffloadConfig
from tests.model_serving.model_server.llmd.utils import (
    get_llmd_inference_pool_pods,
    ns_from_file,
    parse_completion_text,
    send_chat_completions,
    workaround_503_no_healthy_upstream,
)
from utilities.resources.llm_inference_service import LLMInferenceService

pytestmark = [pytest.mark.llmd_gpu]

NAMESPACE = ns_from_file(file=__file__)


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llmisvc",
    [pytest.param({"name": NAMESPACE}, KvCacheCpuOffloadConfig, id="cpu-offload")],
    indirect=True,
)
@pytest.mark.usefixtures("skip_if_disconnected")
class TestLlmdSinglenodeKvCacheCpuOffload:
    """Deploy TinyLlama on GPU with KV cache CPU offloading and verify inference succeeds.

    Note: If kserve generates invalid --kv-transfer-config parameters, vLLM rejects them
    at startup and the pod never becomes Ready — so a successful inference response
    is sufficient proof that the controller produced a valid OffloadingConnector config.
    """

    def test_inference(self, llmisvc: LLMInferenceService):
        """Verify inference succeeds after vLLM starts with CPU KV cache offloading."""
        prompt = "What is the capital of Italy?"
        expected = "rome"

        workaround_503_no_healthy_upstream(llmisvc=llmisvc, prompt=prompt)

        status, body = send_chat_completions(llmisvc=llmisvc, prompt=prompt)
        assert status == 200, f"Expected 200, got {status}: {body}"
        completion = parse_completion_text(response_body=body)
        assert expected in completion.lower(), f"Expected '{expected}' in response, got: {completion}"


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llmisvc",
    [pytest.param({"name": NAMESPACE}, KvCacheDiskOffloadConfig, id="disk-offload")],
    indirect=True,
)
@pytest.mark.usefixtures("skip_if_disconnected")
class TestLlmdSinglenodeKvCacheDiskOffload:
    """Deploy TinyLlama on GPU with a filesystem secondary KV cache tier.

    Validates that the controller attaches an emptyDir-backed ephemeral volume,
    requests matching ephemeral-storage on the container, and that vLLM starts
    and serves requests successfully.
    """

    def test_disk_volume_attached(
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
            expected_vol = KvCacheDiskOffloadConfig.disk_volume_name
            assert expected_vol in volume_names, (
                f"Pod {pod.name}: expected ephemeral volume '{expected_vol}'; got {volume_names}"
            )

            ephemeral_vol = next(v for v in spec.volumes if v.name == expected_vol)
            assert (
                getattr(ephemeral_vol, "emptyDir", None) is not None
                or getattr(ephemeral_vol, "ephemeral", None) is not None
            ), f"Pod {pod.name}: volume '{expected_vol}' is not a local storage volume (got {dict(ephemeral_vol)})"

            container = next(
                (c for c in (spec.containers or []) if c.name == KvCacheDiskOffloadConfig.container_name),
                None,
            )
            assert container is not None, (
                f"Pod {pod.name}: container '{KvCacheDiskOffloadConfig.container_name}' not found"
            )

            mount_paths = [m.mountPath for m in (container.volumeMounts or [])]
            assert KvCacheDiskOffloadConfig.disk_mount_path in mount_paths, (
                f"Pod {pod.name}: expected mount at '{KvCacheDiskOffloadConfig.disk_mount_path}'; got {mount_paths}"
            )

            requests = dict(container.resources.requests or {}) if container.resources else {}
            assert "ephemeral-storage" in requests, (
                f"Pod {pod.name}: expected ephemeral-storage resource request; got {requests}"
            )

    def test_inference(self, llmisvc: LLMInferenceService):
        """Verify inference succeeds with the secondary disk KV cache tier active."""
        prompt = "What is the capital of Italy?"
        expected = "rome"

        workaround_503_no_healthy_upstream(llmisvc=llmisvc, prompt=prompt)

        status, body = send_chat_completions(llmisvc=llmisvc, prompt=prompt)
        assert status == 200, f"Expected 200, got {status}: {body}"
        completion = parse_completion_text(response_body=body)
        assert expected in completion.lower(), f"Expected '{expected}' in response, got: {completion}"
