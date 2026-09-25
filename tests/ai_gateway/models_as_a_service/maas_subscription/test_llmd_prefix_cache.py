"""MaaS Gateway regression coverage for LLM-d prefix-cache endpoint pickers."""

from __future__ import annotations

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.prometheus import Prometheus

from tests.ai_gateway.models_as_a_service.maas_subscription.utils import MaaSLlmdScenario, maas_body_routed_model_name
from tests.model_serving.model_server.llmd.llmd_configs import (  # noqa: NIT001
    EstimatedPrefixCacheConfig,
    PrecisePrefixCacheProducerConfig,
    PrecisePrefixCacheScorerConfig,
)
from tests.model_serving.model_server.llmd.utils import (  # noqa: NIT001
    assert_prefix_cache_routing,
    assert_scheduler_routing,
    get_llmd_inference_pool_pods,
    get_llmd_router_scheduler_pod,
    get_llmd_vllm_pods,
    send_prefix_cache_requests,
)
from utilities.plugins.constant import OpenAIEnpoints

NUM_REQUESTS = 12
PREFIX_CACHE_PROMPT = (
    "Explain in detail the fundamental principles of quantum mechanics including "
    "wave-particle duality, superposition, and entanglement in simple terms. "
    "Additionally, describe how these quantum phenomena differ from classical physics "
    "and why they are important for understanding the nature of reality at the atomic scale."
)


@pytest.mark.tier3
@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.llmd_gpu
@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
)
@pytest.mark.parametrize(
    "maas_llmd_scenario",
    (
        pytest.param(EstimatedPrefixCacheConfig, id="estimated"),
        pytest.param(PrecisePrefixCacheScorerConfig, id="precise-scorer"),
        pytest.param(PrecisePrefixCacheProducerConfig, id="precise-producer"),
    ),
    indirect=True,
)
class TestMaaSGatewayLlmdPrefixCache:
    """Verify EPP-backed prefix-cache routing after MaaS body-based model extraction."""

    def test_prefix_cache_epp_routes_body_routed_requests(
        self,
        request: pytest.FixtureRequest,
        api_key_bound_to_llmd_scenario: str,
        maas_llmd_scenario: MaaSLlmdScenario,
        model_url_llmd_scenario: str,
        prometheus: Prometheus,
        admin_client: DynamicClient,
    ) -> None:
        """Given an EPP-backed prefix-cache LLM-d scenario registered with MaaS,
        when identical prompts identify the model in MaaS Gateway request bodies,
        then the EPP selects a cached backend and records every routing decision.
        """
        llmisvc = maas_llmd_scenario.llmisvc
        config_cls = request.node.callspec.params["maas_llmd_scenario"]
        router_pod = get_llmd_router_scheduler_pod(client=admin_client, llmisvc=llmisvc)
        assert router_pod is not None, "Expected the LLM-d router-scheduler pod"
        assert router_pod.instance.status.phase == "Running", "Expected the LLM-d router-scheduler pod to be Running"

        vllm_pods = get_llmd_vllm_pods(client=admin_client, llmisvc=llmisvc)
        inference_pool_pods = get_llmd_inference_pool_pods(client=admin_client, llmisvc=llmisvc)
        assert len(vllm_pods) == config_cls.expected_vllm_pod_count, (
            f"Expected {config_cls.expected_vllm_pod_count} vLLM pods, found {len(vllm_pods)}"
        )
        assert len(inference_pool_pods) == config_cls.expected_inference_pool_pod_count, (
            f"Expected {config_cls.expected_inference_pool_pod_count} InferencePool pods, "
            f"found {len(inference_pool_pods)}"
        )

        successful_requests = send_prefix_cache_requests(
            llmisvc=llmisvc,
            prompt=PREFIX_CACHE_PROMPT,
            token=api_key_bound_to_llmd_scenario,
            count=NUM_REQUESTS,
            delay_after_first_request=15 if config_cls is not EstimatedPrefixCacheConfig else None,
            inference_base_url=model_url_llmd_scenario.removesuffix(OpenAIEnpoints.CHAT_COMPLETIONS),
            insecure=True,
            request_model_name=maas_body_routed_model_name(llmisvc=llmisvc),
        )

        assert_prefix_cache_routing(
            prometheus=prometheus,
            llmisvc=llmisvc,
            pods=inference_pool_pods,
            expected_requests=successful_requests,
            block_size=config_cls.block_size,
        )
        assert_scheduler_routing(router_pod=router_pod, min_decisions=successful_requests)
