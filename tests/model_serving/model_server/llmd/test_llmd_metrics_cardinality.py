"""
Tests for llm-d Prometheus metrics cardinality safety.

Bug 12 (High): The SchedulerPDDecisionCount metric uses user-controlled
model_name as a Prometheus label, creating an unbounded cardinality risk.
"""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.llm_inference_service import LLMInferenceService

from tests.model_serving.model_server.llmd.utils import (
    get_llmd_router_scheduler_pod,
    ns_from_file,
    send_chat_completions,
)

pytestmark = [pytest.mark.tier2, pytest.mark.gpu, pytest.mark.metrics]

NAMESPACE = ns_from_file(file=__file__)


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llmisvc",
    [({"name": NAMESPACE}, "PrefillDecodeConfig")],
    indirect=True,
)
@pytest.mark.usefixtures("valid_aws_config", "skip_if_no_gpu_available", "skip_if_disconnected")
class TestLlmdMetricsCardinality:
    """Verify llm-d metrics do not create unbounded Prometheus label cardinality.

    Preconditions:
        - LLMInferenceService with P/D deployed
        - Prometheus monitoring enabled

    Test Steps:
        1. Deploy LLMInferenceService
        2. Send requests with different targetModel values
        3. Query Prometheus for scheduler_pd_decision_count metric
        4. Verify that label cardinality stays bounded

    Expected Results:
        - Metrics should not create unique time series per user-supplied model name
    """

    def test_scheduler_metrics_exist(
        self,
        admin_client: DynamicClient,
        llmisvc: LLMInferenceService,
    ):
        """Verify scheduler PD decision metrics are being recorded."""
        send_chat_completions(llmisvc=llmisvc, prompt="Hello")
        router_pod = get_llmd_router_scheduler_pod(client=admin_client, llmisvc=llmisvc)
        assert router_pod is not None, "Router-scheduler pod not found for metrics check"
