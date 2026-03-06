"""GPU inference without scheduler for LLMInferenceService."""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.llm_inference_service import LLMInferenceService

from tests.model_serving.model_server.llmd_v2.llmd_configs import QwenS3Config
from tests.model_serving.model_server.llmd_v2.utils import (
    assert_pods_healthy,
    ns_from_file,
    parse_completion_text,
    send_chat_completions,
    wait_for_llmisvc,
)

pytestmark = [pytest.mark.tier2, pytest.mark.llmd_gpu]

NAMESPACE = ns_from_file(__file__)


class S3GpuNoSchedulerConfig(QwenS3Config):
    name = "llm-gpu-no-scheduler"

    @classmethod
    def router_config(cls):
        return {"route": {}}


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llmisvc",
    [({"name": NAMESPACE}, S3GpuNoSchedulerConfig)],
    indirect=True,
)
@pytest.mark.usefixtures("valid_aws_config")
class TestLlmdNoScheduler:
    """Verify GPU inference works without a scheduler."""

    def test_llmd_no_scheduler(
        self,
        unprivileged_client: DynamicClient,
        llmisvc: LLMInferenceService,
        gpu_count_on_cluster: int,
    ):
        if gpu_count_on_cluster < 1:
            pytest.skip("No GPUs available on cluster, skipping GPU test")

        wait_for_llmisvc(llmisvc)
        assert_pods_healthy(client=unprivileged_client, llmisvc=llmisvc)

        prompt = "What is the capital of Italy?"
        expected = "rome"

        status, body = send_chat_completions(llmisvc, prompt=prompt)
        assert status == 200, f"Expected 200, got {status}: {body}"
        completion = parse_completion_text(body)
        assert expected in completion.lower(), f"Expected '{expected}' in response, got: {completion}"
