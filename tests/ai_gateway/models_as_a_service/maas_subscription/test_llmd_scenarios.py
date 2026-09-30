"""MaaS Gateway data-plane coverage for every reusable LLM-d deployment configuration."""

from __future__ import annotations

import pytest
import requests

from tests.ai_gateway.models_as_a_service.maas_subscription.utils import (
    MaaSLlmdScenario,
    maas_body_routed_model_name,
    poll_expected_status,
)
from tests.ai_gateway.models_as_a_service.utils import build_maas_headers
from tests.model_serving.model_server.llmd.api_compat import (  # noqa: NIT001
    BearerTokenProvider,
    OpenAICompatibilityValidator,
)
from tests.model_serving.model_server.llmd.llmd_configs import (  # noqa: NIT001
    EstimatedPrefixCacheConfig,
    KvCacheCpuOffloadConfig,
    KvCacheDiskOffloadConfig,
    MultinodeMoeDpEpConfig,
    MultinodeMoeDpEpPrefillDecodeConfig,
    PrecisePrefixCacheProducerConfig,
    PrecisePrefixCacheScorerConfig,
    SingleNodePDFast1Config,
    SingleNodePDFast2Config,
    SingleNodePrefillDecodeConfig,
    TinyLlamaFast1Config,
    TinyLlamaFast2Config,
    TinyLlamaHfConfig,
    TinyLlamaHfGpuConfig,
    TinyLlamaOciConfig,
    TinyLlamaOciGpuConfig,
    TinyLlamaS3Config,
    TinyLlamaS3GpuConfig,
    TinyLlamaS3GpuNoSchedulerConfig,
)
from utilities.plugins.constant import OpenAIEnpoints

REQUEST_TIMEOUT_SECONDS = 180

GPU_SCENARIO_MARKS = (pytest.mark.gpu, pytest.mark.llmd_gpu)

LLMD_MAAS_SCENARIOS = (
    pytest.param(TinyLlamaOciConfig, id="cpu-oci"),
    pytest.param(TinyLlamaS3Config, id="cpu-s3"),
    pytest.param(TinyLlamaHfConfig, id="cpu-huggingface"),
    pytest.param(TinyLlamaOciGpuConfig, marks=GPU_SCENARIO_MARKS, id="gpu-oci"),
    pytest.param(TinyLlamaS3GpuConfig, marks=GPU_SCENARIO_MARKS, id="gpu-s3"),
    pytest.param(TinyLlamaS3GpuNoSchedulerConfig, marks=GPU_SCENARIO_MARKS, id="gpu-no-scheduler"),
    pytest.param(TinyLlamaHfGpuConfig, marks=GPU_SCENARIO_MARKS, id="gpu-huggingface"),
    pytest.param(TinyLlamaFast1Config, marks=GPU_SCENARIO_MARKS, id="gpu-fast-1"),
    pytest.param(TinyLlamaFast2Config, marks=GPU_SCENARIO_MARKS, id="gpu-fast-2"),
    pytest.param(EstimatedPrefixCacheConfig, marks=GPU_SCENARIO_MARKS, id="estimated-prefix-cache"),
    pytest.param(PrecisePrefixCacheScorerConfig, marks=GPU_SCENARIO_MARKS, id="precise-prefix-scorer"),
    pytest.param(PrecisePrefixCacheProducerConfig, marks=GPU_SCENARIO_MARKS, id="precise-prefix-producer"),
    pytest.param(SingleNodePrefillDecodeConfig, marks=GPU_SCENARIO_MARKS, id="prefill-decode"),
    pytest.param(SingleNodePDFast1Config, marks=GPU_SCENARIO_MARKS, id="prefill-decode-fast-1"),
    pytest.param(SingleNodePDFast2Config, marks=GPU_SCENARIO_MARKS, id="prefill-decode-fast-2"),
    pytest.param(KvCacheCpuOffloadConfig, marks=GPU_SCENARIO_MARKS, id="kv-cache-cpu"),
    pytest.param(KvCacheDiskOffloadConfig, marks=GPU_SCENARIO_MARKS, id="kv-cache-disk"),
    pytest.param(MultinodeMoeDpEpConfig, marks=GPU_SCENARIO_MARKS, id="multinode-moe-dp-ep"),
    pytest.param(MultinodeMoeDpEpPrefillDecodeConfig, marks=GPU_SCENARIO_MARKS, id="multinode-moe-dp-ep-pd"),
)


@pytest.mark.tier3
@pytest.mark.slow
@pytest.mark.order(1)
@pytest.mark.usefixtures(
    "maas_subscription_controller_enabled_latest",
)
@pytest.mark.parametrize("maas_llmd_scenario", LLMD_MAAS_SCENARIOS, indirect=True)
class TestMaaSGatewayLlmdScenarios:
    """Verify every reusable LLM-d deployment scenario serves body-routed traffic through MaaS."""

    def test_authorized_body_routed_inference(
        self,
        request_session_http: requests.Session,
        api_key_bound_to_llmd_scenario: str,
        maas_llmd_scenario: MaaSLlmdScenario,
        model_url_llmd_scenario: str,
    ) -> None:
        """Given an authorized MaaS subscription for an LLM-d scenario,
        when a chat-completions request identifies the model in its body,
        then the MaaS Gateway forwards it to that scenario successfully.
        """
        llmisvc = maas_llmd_scenario.llmisvc
        response = poll_expected_status(
            request_session_http=request_session_http,
            model_url=model_url_llmd_scenario,
            headers=build_maas_headers(token=api_key_bound_to_llmd_scenario),
            payload={
                "model": maas_body_routed_model_name(llmisvc=llmisvc),
                "messages": [{"role": "user", "content": "Explain request routing in one sentence."}],
                "max_tokens": 32,
            },
            expected_statuses={200},
            request_timeout=REQUEST_TIMEOUT_SECONDS,
        )
        assert response.status_code == 200, (
            f"Expected MaaS-routed inference for '{llmisvc.name}' to succeed, "
            f"got {response.status_code}: {response.text[:200]}"
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
        pytest.param(TinyLlamaFast1Config, id="fast-1"),
        pytest.param(TinyLlamaFast2Config, id="fast-2"),
        pytest.param(EstimatedPrefixCacheConfig, id="estimated-prefix-cache"),
        pytest.param(PrecisePrefixCacheScorerConfig, id="precise-prefix-scorer"),
        pytest.param(PrecisePrefixCacheProducerConfig, id="precise-prefix-producer"),
        pytest.param(SingleNodePrefillDecodeConfig, id="prefill-decode"),
        pytest.param(SingleNodePDFast1Config, id="prefill-decode-fast-1"),
        pytest.param(SingleNodePDFast2Config, id="prefill-decode-fast-2"),
    ),
    indirect=True,
)
class TestMaaSGatewayLlmdOpenAICompatibility:
    """Verify direct LLM-d OpenAI-compatibility scenarios through MaaS."""

    # Run compatibility after the base MaaS inference matrix. The two matrices
    # have intentionally different scenario IDs, so an ``after=<test name>``
    # dependency would leave most parametrized nodes unresolved.
    @pytest.mark.soak
    @pytest.mark.order(2)
    @pytest.mark.parametrize("verification", OpenAICompatibilityValidator.ALL_VERIFICATIONS)
    def test_openai_compatibility(
        self,
        verification: str,
        api_key_bound_to_llmd_scenario: str,
        maas_llmd_scenario: MaaSLlmdScenario,
        model_url_llmd_scenario: str,
    ) -> None:
        """Given a MaaS-authorized LLM-d scenario with tool parsing enabled,
        when an OpenAI validation targets its body-routed Gateway endpoint,
        then that OpenAI or tool-calling assertion passes.
        """
        with OpenAICompatibilityValidator(
            base_url=model_url_llmd_scenario.removesuffix(OpenAIEnpoints.CHAT_COMPLETIONS),
            model_name=maas_body_routed_model_name(llmisvc=maas_llmd_scenario.llmisvc),
            api_key_provider=BearerTokenProvider(token=api_key_bound_to_llmd_scenario),
            verify_ssl=False,
        ) as validator:
            getattr(validator, verification)()
