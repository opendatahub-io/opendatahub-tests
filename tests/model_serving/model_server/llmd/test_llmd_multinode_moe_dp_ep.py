import pytest
from kubernetes.dynamic import DynamicClient

from tests.model_serving.model_server.llmd.llmd_configs import MultinodeMoeDpEpConfig
from tests.model_serving.model_server.llmd.utils import (
    get_llmd_inference_pool_pods,
    get_llmd_router_scheduler_pod,
    get_llmd_vllm_pods,
    ns_from_file,
    parse_completion_text,
    send_chat_completions,
)
from utilities.resources.llm_inference_service import LLMInferenceService

NAMESPACE = ns_from_file(file=__file__)

pytestmark = [pytest.mark.llmd_gpu]


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llmisvc",
    [pytest.param({"name": NAMESPACE}, MultinodeMoeDpEpConfig, id="dp-ep")],
    indirect=True,
)
class TestMultinodeMoeDpEp:
    """Deploy a MoE model across 2 GPU nodes with data parallelism + expert parallelism.

    The controller creates a LeaderWorkerSet with a head pod (template) and worker
    pods (worker). data=2 distributes inference across the nodes, expert=True enables
    MoE expert parallelism.
    """

    def test_vllm_pod_count(
        self,
        request: pytest.FixtureRequest,
        unprivileged_client: DynamicClient,
        llmisvc: LLMInferenceService,
    ) -> None:
        """Test steps:

        1. Get all vLLM pods (leader + workers) for the LLMInferenceService.
        2. Assert the count matches the expected number from the config.
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
    ) -> None:
        """Test steps:

        1. Get pods matching the InferencePool selector (kserve.io/component=workload).
        2. Assert the count matches the expected number from the config.
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
    ) -> None:
        """Test steps:

        1. Get the router-scheduler pod for the LLMInferenceService.
        2. Assert the pod exists.
        3. Assert the pod phase is Running.
        """
        router_pod = get_llmd_router_scheduler_pod(client=unprivileged_client, llmisvc=llmisvc)
        assert router_pod is not None, "Router-scheduler pod should exist"
        assert router_pod.instance.status.phase == "Running", "Router-scheduler pod should be running"

    def test_role_labels(
        self,
        unprivileged_client: DynamicClient,
        llmisvc: LLMInferenceService,
    ) -> None:
        """Test steps:

        1. Get all vLLM pods for the LLMInferenceService.
        2. Assert the leader pod (kserve.io/component=workload) has llm-d.ai/role=both.
        3. Assert worker pods (without kserve.io/component) do not have llm-d.ai/role label.
        """
        vllm_pods = get_llmd_vllm_pods(client=unprivileged_client, llmisvc=llmisvc)

        leaders = []
        workers = []
        for pod in vllm_pods:
            labels = pod.instance.metadata.labels or {}
            if labels.get("kserve.io/component") == "workload":
                leaders.append(pod)
            else:
                workers.append(pod)

        assert len(leaders) >= 1, (
            f"Expected at least 1 leader pod with kserve.io/component=workload, found {len(leaders)}"
        )

        for leader in leaders:
            role = leader.instance.metadata.labels.get("llm-d.ai/role")
            assert role == "both", f"Leader pod {leader.name} expected llm-d.ai/role=both, got: {role!r}"

        for worker in workers:
            labels = worker.instance.metadata.labels or {}
            assert "llm-d.ai/role" not in labels, (
                f"Worker pod {worker.name} should not have llm-d.ai/role label, got: {labels.get('llm-d.ai/role')!r}"
            )

    def test_multinode_spread(
        self,
        request: pytest.FixtureRequest,
        unprivileged_client: DynamicClient,
        llmisvc: LLMInferenceService,
    ) -> None:
        """Test steps:

        1. Get all vLLM pods for the LLMInferenceService.
        2. Extract the node name from each pod.
        3. Assert pods are spread across at least min_nodes distinct nodes.
        """
        config = request.node.callspec.params["llmisvc"]
        vllm_pods = get_llmd_vllm_pods(client=unprivileged_client, llmisvc=llmisvc)
        unique_nodes = {pod.instance.spec.nodeName for pod in vllm_pods}
        assert len(unique_nodes) >= config.min_nodes, (
            f"Expected pods on >= {config.min_nodes} nodes, found {len(unique_nodes)}: {unique_nodes}"
        )

    def test_workers_excluded_from_pool(
        self,
        unprivileged_client: DynamicClient,
        llmisvc: LLMInferenceService,
    ) -> None:
        """Test steps:

        1. Get all vLLM pods and InferencePool pods.
        2. Identify worker pods (vLLM pods not in the pool).
        3. Assert each worker pod does NOT have label kserve.io/component.
        4. Assert each worker pod does NOT have label llm-d.ai/role.
        """
        vllm_pods = get_llmd_vllm_pods(client=unprivileged_client, llmisvc=llmisvc)
        pool_pods = get_llmd_inference_pool_pods(client=unprivileged_client, llmisvc=llmisvc)
        pool_pod_names = {pod.name for pod in pool_pods}
        worker_pods = [pod for pod in vllm_pods if pod.name not in pool_pod_names]

        assert worker_pods, "Expected at least one worker pod not in the InferencePool"

        for pod in worker_pods:
            labels = pod.instance.metadata.labels or {}
            assert "kserve.io/component" not in labels, (
                f"Worker pod {pod.name} should NOT have label kserve.io/component,"
                f" got: {labels.get('kserve.io/component')}"
            )
            assert "llm-d.ai/role" not in labels, (
                f"Worker pod {pod.name} should NOT have label llm-d.ai/role, got: {labels.get('llm-d.ai/role')}"
            )

    def test_inference(
        self,
        llmisvc: LLMInferenceService,
    ) -> None:
        """Test steps:

        1. Send a chat completion request to /v1/chat/completions.
        2. Assert the response status is 200.
        3. Assert the completion text is non-empty.
        """
        status_code, response_body = send_chat_completions(
            llmisvc=llmisvc,
            prompt="This model reply with garbage completion.",
        )
        assert status_code == 200, f"Expected 200, got {status_code}: {response_body}"
        completion = parse_completion_text(response_body=response_body)
        assert completion.strip(), f"Expected non-empty completion, got: {completion!r}"
