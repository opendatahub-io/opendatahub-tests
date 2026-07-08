import re

import pytest
from ocp_resources.llm_inference_service import LLMInferenceService

from tests.model_serving.model_server.llmd.llmd_configs import TinyLlamaFast1Config, TinyLlamaFast2Config
from tests.model_serving.model_server.llmd.utils import (
    get_vllm_version,
    ns_from_file,
    parse_completion_text,
    send_chat_completions,
    workaround_503_no_healthy_upstream,
)

pytestmark = [pytest.mark.tier1, pytest.mark.llmd_gpu, pytest.mark.fast_vllm]

NAMESPACE = ns_from_file(file=__file__)


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llmisvc",
    [
        pytest.param({"name": NAMESPACE}, TinyLlamaFast1Config, id="test_fast_1"),
        pytest.param({"name": NAMESPACE}, TinyLlamaFast2Config, id="test_fast_2"),
    ],
    indirect=True,
)
class TestLlmdFastImage:
    """Deploy TinyLlama using fast-1 and fast-2 LLMInferenceServiceConfig and verify inference."""

    def test_inference(
        self,
        llmisvc: LLMInferenceService,
    ) -> None:
        """Verify chat completions return a valid response using a fast image.

        Given a model deployed via a fast LLMInferenceServiceConfig,
        When a chat completion request is sent to /v1/chat/completions,
        Then the response status is 200 and the completion text contains the expected answer.
        """
        prompt = "What is the capital of Italy?"
        expected = "rome"

        workaround_503_no_healthy_upstream(llmisvc=llmisvc, prompt=prompt)

        status, body = send_chat_completions(llmisvc=llmisvc, prompt=prompt)
        assert status == 200, f"Expected 200, got {status}: {body}"
        completion = parse_completion_text(response_body=body)
        assert expected in completion.lower(), f"Expected '{expected}' in response, got: {completion}"

    def test_vllm_version(
        self,
        llmisvc: LLMInferenceService,
    ) -> None:
        """Verify the served model endpoint reports a valid vLLM version.

        Given a model deployed via a fast LLMInferenceServiceConfig,
        When querying the /version endpoint,
        Then the response contains a non-empty vLLM version string.
        """
        version = get_vllm_version(llmisvc=llmisvc)
        assert version, "Expected a non-empty vLLM version string from /version endpoint"
        assert re.fullmatch(r"\d+\.\d+\.\d+(?:-[a-zA-Z0-9.]+)?(?:\+[a-zA-Z0-9.]+)?", version), (
            f"vLLM version '{version}' does not match expected semver format (e.g. '0.8.5')"
        )
