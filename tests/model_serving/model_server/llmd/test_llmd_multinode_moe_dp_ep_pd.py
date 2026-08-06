import pytest
from kubernetes.dynamic import DynamicClient

from tests.model_serving.model_server.llmd.llmd_configs import MultinodeMoeDpEPPrefillDecodeConfig
from tests.model_serving.model_server.llmd.utils import (
    get_llmd_inference_pool_pods,
    get_llmd_pod_by_role,
    get_llmd_router_scheduler_pod,
    get_llmd_vllm_pods,
    ns_from_file,
    parse_completion_text,
    scheduler_has_plugin,
    send_chat_completions,
)
from utilities.resources.llm_inference_service import LLMInferenceService

NAMESPACE = ns_from_file(file=__file__)

pytestmark = [pytest.mark.llmd_gpu]


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llmisvc",
    [pytest.param({"name": NAMESPACE}, MultinodeMoeDpEPPrefillDecodeConfig, id="dp-ep-pd")],
    indirect=True,
)
class TestMultinodeMoeDpEPPrefillDecode:
    """Multinode MoE with DP+EP parallelism and disaggregated Prefill/Decode.

    Validates that combining multinode data-parallel + expert-parallel with P/D
    disaggregation produces the correct topology: decode and prefill LWS groups
    spanning multiple nodes, with controller-generated scheduler plugins for P/D
    routing.
    """

    def test_vllm_pod_count(
        self,
        request: pytest.FixtureRequest,
        unprivileged_client: DynamicClient,
        llmisvc: LLMInferenceService,
    ):
        """Test steps:

        1. Get all vLLM pods (decode + prefill, leaders + workers).
        2. Assert the count matches expected (decode LWS + prefill LWS).
        """
        config = request.node.callspec.params["llmisvc"]
        vllm_pods = get_llmd_vllm_pods(client=unprivileged_client, llmisvc=llmisvc)
        assert len(vllm_pods) == config.expected_vllm_pod_count, (
            f"Expected {config.expected_vllm_pod_count} vLLM pods, found {len(vllm_pods)}"
        )

    def test_inference_pool_pod_count(
        self,
        request: pytest.FixtureRequest,
        unprivileged_client: DynamicClient,
        llmisvc: LLMInferenceService,
    ):
        """Test steps:

        1. Get pods matching the InferencePool selector.
        2. Assert only decode pods are pool members.
        """
        config = request.node.callspec.params["llmisvc"]
        inferencepool_pods = get_llmd_inference_pool_pods(client=unprivileged_client, llmisvc=llmisvc)
        assert len(inferencepool_pods) == config.expected_inference_pool_pod_count, (
            f"Expected {config.expected_inference_pool_pod_count} InferencePool pods, found {len(inferencepool_pods)}"
        )

    def test_router_scheduler(
        self,
        unprivileged_client: DynamicClient,
        llmisvc: LLMInferenceService,
    ):
        """Test steps:

        1. Get the router-scheduler pod.
        2. Assert it exists and is Running.
        """
        router_pod = get_llmd_router_scheduler_pod(client=unprivileged_client, llmisvc=llmisvc)
        assert router_pod is not None, "Router-scheduler pod should exist"
        assert router_pod.instance.status.phase == "Running", "Router-scheduler pod should be running"

    def test_prefill_decode_topology(
        self,
        unprivileged_client: DynamicClient,
        llmisvc: LLMInferenceService,
    ):
        """Test steps:

        1. Get all vLLM pods and inspect llm-d.ai/role labels.
        2. Assert both 'decode' and 'prefill' roles exist.
        3. Assert decode pods have llm-d-routing-sidecar init container.
        4. Assert prefill pods do not have the sidecar.
        """
        vllm_pods = get_llmd_vllm_pods(client=unprivileged_client, llmisvc=llmisvc)

        roles = {}
        for pod in vllm_pods:
            role = pod.instance.metadata.labels.get("llm-d.ai/role")
            if role:
                roles.setdefault(role, []).append(pod.name)

        assert sorted(roles.keys()) == ["decode", "prefill"], (
            f"Expected 'decode' and 'prefill' roles, got: {dict(roles)}"
        )

        decode_pod = get_llmd_pod_by_role(client=unprivileged_client, llmisvc=llmisvc, role="decode")
        init_containers = [c.name for c in (decode_pod.instance.spec.get("initContainers") or [])]
        assert "llm-d-routing-sidecar" in init_containers, (
            f"Decode pod missing llm-d-routing-sidecar init container, found: {init_containers}"
        )

        prefill_pod = get_llmd_pod_by_role(client=unprivileged_client, llmisvc=llmisvc, role="prefill")
        prefill_init_containers = [c.name for c in (prefill_pod.instance.spec.get("initContainers") or [])]
        assert "llm-d-routing-sidecar" not in prefill_init_containers, (
            f"Prefill pod should NOT have llm-d-routing-sidecar, found: {prefill_init_containers}"
        )

    def test_scheduler_pd_plugins(
        self,
        unprivileged_client: DynamicClient,
        llmisvc: LLMInferenceService,
    ):
        """Test steps:

        1. Check the scheduler config for all expected P/D plugins.
        2. Assert all 5 disaggregation plugins are present.
        """
        for expected_plugin in [
            "disagg-headers-handler",
            "prefill-filter",
            "decode-filter",
            "always-disagg-pd-decider",
            "disagg-profile-handler",
        ]:
            assert scheduler_has_plugin(client=unprivileged_client, llmisvc=llmisvc, plugin_name=expected_plugin), (
                f"Scheduler config missing expected P/D plugin: {expected_plugin}"
            )

    def test_inference(
        self,
        llmisvc: LLMInferenceService,
    ):
        """Test steps:

        1. Send a chat completion request.
        2. Assert HTTP 200 and non-empty completion.
        """
        status_code, response_body = send_chat_completions(
            llmisvc=llmisvc,
            prompt="This model reply with garbage completion.",
        )
        assert status_code == 200, f"Expected 200, got {status_code}: {response_body}"
        completion = parse_completion_text(response_body=response_body)
        assert completion.strip(), f"Expected non-empty completion, got: {completion!r}"
