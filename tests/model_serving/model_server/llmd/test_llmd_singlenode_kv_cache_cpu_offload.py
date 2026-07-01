import pytest
from ocp_resources.llm_inference_service import LLMInferenceService

from tests.model_serving.model_server.llmd.llmd_configs import KvCacheCpuOffloadConfig
from tests.model_serving.model_server.llmd.utils import (
    ns_from_file,
    parse_completion_text,
    send_chat_completions,
    workaround_503_no_healthy_upstream,
)

pytestmark = [pytest.mark.smoke, pytest.mark.llmd_gpu]

NAMESPACE = ns_from_file(file=__file__)


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llmisvc",
    [({"name": NAMESPACE}, KvCacheCpuOffloadConfig)],
    indirect=True,
)
@pytest.mark.usefixtures("skip_if_disconnected")
class TestLlmdSinglenodeKvCacheCpuOffload:
    """Deploy TinyLlama on GPU with KV cache CPU offloading and verify inference succeeds.

    If kserve generates invalid --kv-transfer-config parameters, vLLM rejects them
    at startup and the pod never becomes Ready — so a successful inference response
    is sufficient proof that the controller produced a valid OffloadingConnector config.
    """

    def test_llmd_singlenode_kv_cache_cpu_offload(
        self,
        llmisvc: LLMInferenceService,
    ):
        """Test steps:

        1. Fixture wait_for_llmisvc_pods_ready ensures vLLM started successfully
           with the --kv-transfer-config parameters kserve injected.
        2. Send a chat completion request and assert a 200 response with the expected answer.
        """
        prompt = "What is the capital of Italy?"
        expected = "rome"

        workaround_503_no_healthy_upstream(llmisvc=llmisvc, prompt=prompt)

        status, body = send_chat_completions(llmisvc=llmisvc, prompt=prompt)
        assert status == 200, f"Expected 200, got {status}: {body}"
        completion = parse_completion_text(response_body=body)
        assert expected in completion.lower(), f"Expected '{expected}' in response, got: {completion}"
