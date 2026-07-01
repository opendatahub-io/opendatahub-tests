import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.llm_inference_service import LLMInferenceService

from tests.model_serving.model_server.llmd.llmd_configs import KvCacheCpuOffloadConfig
from tests.model_serving.model_server.llmd.utils import (
    get_llmd_workload_pods,
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
    """Deploy TinyLlama on GPU with KV cache CPU offloading and verify inference and connector config."""

    def test_llmd_singlenode_kv_cache_cpu_offload(
        self,
        unprivileged_client: DynamicClient,
        llmisvc: LLMInferenceService,
    ):
        """Test steps:

        1. Assert exactly 1 workload pod is Running.
        2. Assert the workload pod logs contain OffloadingConnector, confirming
           the controller injected --kv-transfer-config correctly.
        3. Send a chat completion request and assert a 200 response with the expected answer.
        """
        workload_pods = get_llmd_workload_pods(client=unprivileged_client, llmisvc=llmisvc)
        assert len(workload_pods) == 1, f"Expected 1 workload pod, found {len(workload_pods)}"

        pod = workload_pods[0]
        assert pod.instance.status.phase == "Running", f"Workload pod {pod.name} is not Running"

        pod_logs = pod.log(container="main")
        assert "OffloadingConnector" in pod_logs, (
            f"Expected 'OffloadingConnector' in pod {pod.name} logs, indicating --kv-transfer-config was applied"
        )

        prompt = "What is the capital of Italy?"
        expected = "rome"

        workaround_503_no_healthy_upstream(llmisvc=llmisvc, prompt=prompt)

        status, body = send_chat_completions(llmisvc=llmisvc, prompt=prompt)
        assert status == 200, f"Expected 200, got {status}: {body}"
        completion = parse_completion_text(response_body=body)
        assert expected in completion.lower(), f"Expected '{expected}' in response, got: {completion}"
