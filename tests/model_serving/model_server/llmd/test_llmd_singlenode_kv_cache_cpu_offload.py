"""E2e smoke test for KV cache CPU offloading on a GPU cluster."""

import pytest

from tests.model_serving.model_server.llmd.llmd_configs import TinyLlamaOciGpuConfig
from tests.model_serving.model_server.llmd.utils import (
    ns_from_file,
    parse_completion_text,
    send_chat_completions,
    workaround_503_no_healthy_upstream,
)
from utilities.resources.llm_inference_service import LLMInferenceService

pytestmark = [pytest.mark.llmd_gpu]

NAMESPACE = ns_from_file(file=__file__)

_KvCacheCpuOffloadConfig = TinyLlamaOciGpuConfig.with_overrides(
    name="llmisvc-kv-cache-cpu-offload",
    kv_cache_offloading={"cpu": "4Gi", "evictionPolicy": "lru"},
)


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llmisvc",
    [({"name": NAMESPACE}, _KvCacheCpuOffloadConfig)],
    indirect=True,
)
@pytest.mark.usefixtures("skip_if_disconnected")
class TestLlmdSinglenodeKvCacheCpuOffload:
    """Deploy TinyLlama on GPU with KV cache CPU offloading and verify inference succeeds.

    Steps:
        1. Deploy TinyLlama on GPU with KV cache CPU offloading configured (4Gi CPU tier, lru eviction policy).
        2. Send a chat completions request to the deployed model.
        3. Verify the response status is HTTP 200 and the response text contains the expected answer.

    Note: If kserve generates invalid --kv-transfer-config parameters, vLLM rejects them
    at startup and the pod never becomes Ready — so a successful inference response
    is sufficient proof that the controller produced a valid OffloadingConnector config.
    """

    def test_llmd_singlenode_kv_cache_cpu_offload(
        self,
        llmisvc: LLMInferenceService,
    ):
        """Verify inference succeeds after vLLM starts with --kv-transfer-config injected."""
        prompt = "What is the capital of Italy?"
        expected = "rome"

        workaround_503_no_healthy_upstream(llmisvc=llmisvc, prompt=prompt)

        status, body = send_chat_completions(llmisvc=llmisvc, prompt=prompt)
        assert status == 200, f"Expected 200, got {status}: {body}"
        completion = parse_completion_text(response_body=body)
        assert expected in completion.lower(), f"Expected '{expected}' in response, got: {completion}"
