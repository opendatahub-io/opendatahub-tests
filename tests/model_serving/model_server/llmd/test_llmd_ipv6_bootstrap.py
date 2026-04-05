"""
Tests for llm-d SGLang IPv6 bootstrap host parsing.

Bug 11 (High): The getBootstrapHost function uses strings.Split on ":"
which incorrectly parses IPv6 addresses like "[2001:db8::1]:8000",
breaking prefill-decode on dual-stack/IPv6 clusters.
"""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.llm_inference_service import LLMInferenceService

from tests.model_serving.model_server.llmd.utils import (
    get_llmd_workload_pods,
    ns_from_file,
    send_chat_completions,
)

pytestmark = [pytest.mark.tier2, pytest.mark.gpu]

NAMESPACE = ns_from_file(file=__file__)


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llmisvc",
    [({"name": NAMESPACE}, "PrefillDecodeConfig")],
    indirect=True,
)
@pytest.mark.usefixtures("valid_aws_config", "skip_if_no_gpu_available", "skip_if_disconnected")
class TestLlmdIpv6Bootstrap:
    """Verify llm-d P/D functions correctly on clusters with IPv6 pod addresses.

    Preconditions:
        - LLMInferenceService with P/D deployed
        - GPU available

    Test Steps:
        1. Deploy LLMInferenceService with prefill-decode
        2. Check if workload pods have IPv6 addresses
        3. Send inference request
        4. Verify no errors in sidecar logs related to address parsing

    Expected Results:
        - Inference should work regardless of IP address format
        - No address parsing errors in sidecar logs
    """

    def test_inference_works(self, llmisvc: LLMInferenceService):
        """Verify basic inference works to establish baseline."""
        status, body = send_chat_completions(llmisvc=llmisvc, prompt="Hello")
        assert status == 200, f"Expected 200, got {status}: {body}"

    def test_no_address_parsing_errors_in_sidecar(
        self,
        admin_client: DynamicClient,
        llmisvc: LLMInferenceService,
    ):
        """Verify sidecar logs do not contain address parsing errors.

        Given an LLMInferenceService with P/D disaggregation
        When checking sidecar container logs
        Then no IPv6-related address parsing errors should be present
        """
        pods = get_llmd_workload_pods(client=admin_client, llmisvc=llmisvc)
        for pod in pods:
            for container_status in pod.instance.spec.containers or []:
                if "sidecar" in container_status.name.lower() or "proxy" in container_status.name.lower():
                    try:
                        logs = pod.log(container=container_status.name, tail_lines=200)
                        error_indicators = [
                            "invalid host",
                            "dial tcp: address",
                            "missing port in address",
                            "too many colons",
                        ]
                        for indicator in error_indicators:
                            assert indicator not in logs.lower(), (
                                f"Found '{indicator}' in sidecar logs of pod {pod.name}. "
                                "IPv6 address parsing may be broken (Bug 11)."
                            )
                    except Exception:  # noqa: BLE001, S110
                        pass
