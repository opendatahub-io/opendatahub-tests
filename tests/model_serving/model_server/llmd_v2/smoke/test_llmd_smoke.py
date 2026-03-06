"""Smoke test for LLMD — OCI CPU inference with TinyLlama."""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.llm_inference_service import LLMInferenceService

from tests.model_serving.model_server.llmd_v2.llmd_configs import TinyLlamaOciConfig
from tests.model_serving.model_server.llmd_v2.utils import (
    assert_pods_healthy,
    ns_from_file,
    parse_completion_text,
    send_chat_completions,
    wait_for_llmisvc,
)

pytestmark = [pytest.mark.smoke]

NAMESPACE = ns_from_file(__file__)


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llmisvc",
    [({"name": NAMESPACE}, TinyLlamaOciConfig)],
    indirect=True,
)
class TestLLMDSmoke:
    """Smoke test: OCI CPU inference with TinyLlama via the config-driven fixture."""

    def test_llmd_smoke(
        self,
        unprivileged_client: DynamicClient,
        llmisvc: LLMInferenceService,
    ):
        """Verify basic LLMD inference works end-to-end."""
        wait_for_llmisvc(llmisvc=llmisvc)
        assert_pods_healthy(client=unprivileged_client, llmisvc=llmisvc)

        prompt = "What is the capital of Italy?"
        expected = "rome"

        status, body = send_chat_completions(llmisvc=llmisvc, prompt=prompt)
        assert status == 200, f"Expected 200, got {status}: {body}"
        completion = parse_completion_text(body)
        assert expected in completion.lower(), f"Expected '{expected}' in response, got: {completion}"
