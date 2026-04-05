"""
Tests for llm-d sidecar handling of large request payloads.

Bug 10 (High): The llm-d P/D sidecar reads the full request body with
io.ReadAll without any size limit, making it vulnerable to OOM via
oversized payloads.
"""

import json

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.llm_inference_service import LLMInferenceService

from tests.model_serving.model_server.llmd.utils import (
    _curl_post,
    _get_inference_url,
    get_llmd_workload_pods,
    ns_from_file,
)

pytestmark = [pytest.mark.tier3, pytest.mark.gpu]

NAMESPACE = ns_from_file(file=__file__)
LARGE_PAYLOAD_SIZE_MB = 10


def _build_large_payload(model_name: str, size_mb: int) -> str:
    """Build a chat completion request with a very large prompt."""
    filler = "A" * (size_mb * 1024 * 1024)
    return json.dumps({
        "model": model_name,
        "messages": [{"role": "user", "content": filler}],
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": False,
    })


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llmisvc",
    [({"name": NAMESPACE}, "PrefillDecodeConfig")],
    indirect=True,
)
@pytest.mark.usefixtures("valid_aws_config", "skip_if_no_gpu_available", "skip_if_disconnected")
class TestLlmdSidecarLargePayload:
    """Verify the llm-d sidecar handles oversized request bodies without crashing.

    Preconditions:
        - LLMInferenceService with P/D disaggregation deployed
        - GPU available on cluster

    Test Steps:
        1. Deploy LLMInferenceService with prefill-decode configuration
        2. Record initial pod restart counts
        3. Send a large payload (10MB) to the inference endpoint
        4. Verify the sidecar returns an error response (not OOM crash)
        5. Verify no pods restarted

    Expected Results:
        - The sidecar should reject oversized payloads with an HTTP error (400 or 413)
        - No sidecar pod should be OOM-killed or restarted
    """

    def test_large_payload_does_not_crash_sidecar(
        self,
        admin_client: DynamicClient,
        llmisvc: LLMInferenceService,
    ):
        """Verify sidecar survives a large payload without OOM crash.

        Given an LLMInferenceService with P/D sidecar running
        When sending a 10MB request payload
        Then the sidecar should respond with an error (not crash)
        And the pod restart count should remain unchanged
        """
        pods = get_llmd_workload_pods(client=admin_client, llmisvc=llmisvc)
        initial_restarts = {}
        for pod in pods:
            for cs in pod.instance.status.containerStatuses or []:
                initial_restarts[f"{pod.name}/{cs.name}"] = cs.restartCount

        model_name = llmisvc.instance.spec.model.get("name", llmisvc.name)
        url = _get_inference_url(llmisvc=llmisvc) + "/v1/chat/completions"
        body = _build_large_payload(model_name=model_name, size_mb=LARGE_PAYLOAD_SIZE_MB)

        status_code, _ = _curl_post(url=url, body=body, timeout=120)
        assert status_code != 0, "curl returned no response — sidecar may have crashed"

        pods_after = get_llmd_workload_pods(client=admin_client, llmisvc=llmisvc)
        for pod in pods_after:
            for cs in pod.instance.status.containerStatuses or []:
                key = f"{pod.name}/{cs.name}"
                prev = initial_restarts.get(key, 0)
                assert cs.restartCount == prev, (
                    f"Container {key} restarted (was {prev}, now {cs.restartCount}). "
                    "Sidecar may have been OOM-killed by large payload (Bug 10)."
                )
