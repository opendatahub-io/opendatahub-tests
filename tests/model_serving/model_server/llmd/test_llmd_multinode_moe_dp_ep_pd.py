import pytest
from kubernetes.dynamic import DynamicClient

from tests.model_serving.model_server.llmd.llmd_configs import MultinodeMoeDpEpPrefillDecodeConfig
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
from utilities.resources.leader_worker_set import LeaderWorkerSet
from utilities.resources.llm_inference_service import LLMInferenceService

NAMESPACE = ns_from_file(file=__file__)

pytestmark = [pytest.mark.llmd_gpu]


@pytest.mark.parametrize(
    "unprivileged_model_namespace, llmisvc",
    [pytest.param({"name": NAMESPACE}, MultinodeMoeDpEpPrefillDecodeConfig, id="dp-ep-pd")],
    indirect=True,
)
class TestMultinodeMoeDpEpPrefillDecode:
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

    def test_lws_count(
        self,
        admin_client: DynamicClient,
        llmisvc: LLMInferenceService,
    ):
        """Test steps:

        1. List all LeaderWorkerSet objects in the namespace.
        2. Assert exactly 2 LWS exist (decode + prefill).
        3. Assert one LWS name ends with '-kserve-mn' (decode) and one ends with '-kserve-mn-prefill' (prefill).
        4. Assert both have readyReplicas matching their spec.replicas.
        """
        lws_list = list(
            LeaderWorkerSet.get(
                client=admin_client,
                namespace=llmisvc.namespace,
            )
        )
        lws_names = sorted(lws.name for lws in lws_list)

        assert len(lws_list) == 2, f"Expected 2 LeaderWorkerSet objects, found {len(lws_list)}: {lws_names}"

        decode_lws = [lws for lws in lws_list if lws.name.endswith("-kserve-mn")]
        prefill_lws = [lws for lws in lws_list if lws.name.endswith("-kserve-mn-prefill")]
        assert len(decode_lws) == 1, (
            f"Expected 1 LWS ending with '-kserve-mn' (decode), found {len(decode_lws)}: {lws_names}"
        )
        assert len(prefill_lws) == 1, (
            f"Expected 1 LWS ending with '-kserve-mn-prefill' (prefill), found {len(prefill_lws)}: {lws_names}"
        )

        for lws in lws_list:
            spec_replicas = lws.instance.spec.get("replicas", 1)
            ready_replicas = lws.instance.get("status", {}).get("readyReplicas", 0)
            assert ready_replicas == spec_replicas, (
                f"LWS {lws.name}: readyReplicas ({ready_replicas}) != spec.replicas ({spec_replicas})"
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

    def test_workers_excluded_from_pool(
        self,
        unprivileged_client: DynamicClient,
        llmisvc: LLMInferenceService,
    ):
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

    def test_multinode_spread(
        self,
        request: pytest.FixtureRequest,
        unprivileged_client: DynamicClient,
        llmisvc: LLMInferenceService,
    ):
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
