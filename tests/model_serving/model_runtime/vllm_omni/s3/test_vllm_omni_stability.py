"""Stability and performance tests for vLLM-Omni TTS models."""

import itertools
import os
import time

import pytest
import structlog
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from ocp_utilities.monitoring import Prometheus

from tests.model_serving.model_runtime.utils import get_restart_counts, pod_is_ready
from tests.model_serving.model_runtime.vllm_omni.constant import (
    E2E_P95_THRESHOLD_S,
    FIFTY_TURN_COUNT,
    OMNI_TTS_VOICE,
    OMNIVOICE_MODEL_PATH,
    PERF_REQUEST_COUNT,
    PROMETHEUS_KV_CACHE_METRIC,
    QWEN3_TTS_MODEL_PATH,
    SOAK_GPU_MEMORY_GROWTH_THRESHOLD,
    SOAK_TTFB_DEGRADATION_RATIO,
    TTFB_P95_THRESHOLD_S,
    TTS_PROMPT_CORPUS,
    VOXTRAL_TTS_MODEL_PATH,
    WARM_UP_COUNT,
)
from tests.model_serving.model_runtime.vllm_omni.utils import (
    assert_no_pod_restarts,
    calculate_percentile,
    log_loop_progress,
    timed_tts_request,
    warmup_tts,
)
from utilities.constants import KServeDeploymentType
from utilities.monitoring import get_metrics_value

LOGGER = structlog.get_logger(name=__name__)

pytestmark = pytest.mark.usefixtures("skip_if_no_supported_accelerator_type", "valid_aws_config")


@pytest.mark.vllm_omni_nvidia_single_gpu
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, vllm_omni_serving_runtime, vllm_omni_inference_service",
    [
        pytest.param(
            {"name": "vllm-omni-tts-stability"},
            {"model-dir": QWEN3_TTS_MODEL_PATH},
            {"deployment_mode": KServeDeploymentType.STANDARD},
            {
                "name": "vllm-omni-tts-stability",
                "model_path": QWEN3_TTS_MODEL_PATH,
                "deployment_mode": KServeDeploymentType.STANDARD,
                "gpu_count": 1,
            },
            id="test_vllm_omni_qwen3_tts_stability",
        ),
    ],
    indirect=True,
)
class TestVllmOmniQwen3TtsStabilityAndPerformance:
    """Stability and latency tests for the Qwen3-TTS model over a shared ISVC.

    Sends 50 sequential TTS requests and verifies the pod does not restart.
    Also sends 200+ requests from the same deployment and asserts TTFB p95
    <= 350 ms and E2E p95 <= 1.2 s.
    """

    def test_vllm_omni_tts_50_turn_session(
        self,
        vllm_omni_inference_service: InferenceService,
        vllm_omni_isvc_url: str,
        vllm_omni_pod_resource: Pod,
    ) -> None:
        """Qwen3-TTS sustains 50 sequential voice turns without restart.

        Given a vLLM-Omni InferenceService with Qwen3-TTS loaded from S3 on 1x GPU,
        When 50 sequential /v1/audio/speech requests are sent using a fixed prompt corpus,
        Then all 50 requests return HTTP 200, response bodies are non-empty, and the
        pod restart count is unchanged from the value recorded before the first request.
        """
        assert pod_is_ready(pod=vllm_omni_pod_resource), (
            f"Pod {vllm_omni_pod_resource.name} must be Ready before starting the 50-turn session"
        )

        initial_restart_counts = get_restart_counts(pod=vllm_omni_pod_resource)
        LOGGER.info(
            event="50-turn TTS stability: starting session",
            pod=vllm_omni_pod_resource.name,
            initial_restart_counts=initial_restart_counts,
        )

        base_url = vllm_omni_isvc_url

        warmup_tts(
            url=base_url,
            model=vllm_omni_inference_service.instance.metadata.name,
            voice=OMNI_TTS_VOICE[QWEN3_TTS_MODEL_PATH],
            count=WARM_UP_COUNT,
        )

        failed_turns: list[int] = []
        for turn_idx in range(FIFTY_TURN_COUNT):
            prompt = TTS_PROMPT_CORPUS[turn_idx % len(TTS_PROMPT_CORPUS)]
            status_code, ttfb_s, e2e_s = timed_tts_request(
                url=base_url,
                model=vllm_omni_inference_service.instance.metadata.name,
                prompt=prompt,
                voice=OMNI_TTS_VOICE[QWEN3_TTS_MODEL_PATH],
            )
            log_loop_progress(
                logger=LOGGER,
                event="50-turn TTS stability: turn result",
                current=turn_idx + 1,
                total=FIFTY_TURN_COUNT,
                is_error=status_code != 200,
                status_code=status_code,
                ttfb_s=round(ttfb_s, 3),
                e2e_s=round(e2e_s, 3),
            )
            if status_code != 200:
                failed_turns.append(turn_idx + 1)

        assert not failed_turns, (
            f"50-turn TTS stability failed: {len(failed_turns)}/50 turns returned non-200 status. "
            f"Failed turn indices: {failed_turns}"
        )

        assert_no_pod_restarts(
            pod=vllm_omni_pod_resource,
            initial_counts=initial_restart_counts,
            context="50-turn TTS stability",
        )

        assert pod_is_ready(pod=vllm_omni_pod_resource), (
            f"Pod {vllm_omni_pod_resource.name} is no longer Ready after the 50-turn session"
        )

    def test_vllm_omni_tts_ttfb_and_e2e_latency(
        self,
        vllm_omni_inference_service: InferenceService,
        vllm_omni_isvc_url: str,
    ) -> None:
        """Qwen3-TTS TTFB p95 <= 350 ms and E2E p95 <= 1.2 s over 200+ requests.

        Given a vLLM-Omni InferenceService with Qwen3-TTS on 1x NVIDIA GPU,
        When 2 warm-up requests are sent and then 200 sequential /v1/audio/speech requests
        are issued using the fixed 50-prompt corpus cycled 4 times,
        Then TTFB p95 is at most TTFB_P95_THRESHOLD_S and E2E p95 is at most
        E2E_P95_THRESHOLD_S, both measured at the OpenShift Route, providing at least
        200 data points for statistical validity.
        """
        base_url = vllm_omni_isvc_url
        LOGGER.info(
            event="TTFB/E2E latency measurement: starting",
            model=vllm_omni_inference_service.instance.metadata.name,
            request_count=PERF_REQUEST_COUNT,
            warm_up_count=WARM_UP_COUNT,
        )

        warmup_tts(
            url=base_url,
            model=vllm_omni_inference_service.instance.metadata.name,
            voice=OMNI_TTS_VOICE[QWEN3_TTS_MODEL_PATH],
            count=WARM_UP_COUNT,
        )

        measurement_corpus = list(
            itertools.islice(  # noqa: FCN001
                itertools.cycle(TTS_PROMPT_CORPUS),
                PERF_REQUEST_COUNT,  # noqa: FCN001
            )
        )

        ttfb_values: list[float] = []
        e2e_values: list[float] = []

        for req_idx, prompt in enumerate(measurement_corpus):
            status_code, ttfb_s, e2e_s = timed_tts_request(
                url=base_url,
                model=vllm_omni_inference_service.instance.metadata.name,
                prompt=prompt,
                voice=OMNI_TTS_VOICE[QWEN3_TTS_MODEL_PATH],
            )
            assert status_code == 200, (
                f"Latency test: request #{req_idx + 1} returned HTTP {status_code} (expected 200)"
            )
            ttfb_values.append(ttfb_s)
            e2e_values.append(e2e_s)
            log_loop_progress(
                logger=LOGGER,
                event="latency measurement: request completed",
                current=req_idx + 1,
                total=PERF_REQUEST_COUNT,
                every=20,
                ttfb_s=round(ttfb_s, 3),
                e2e_s=round(e2e_s, 3),
            )

        assert len(ttfb_values) >= PERF_REQUEST_COUNT, (
            f"Latency test: collected only {len(ttfb_values)} measurements; "
            f"need at least {PERF_REQUEST_COUNT} for statistical validity"
        )

        ttfb_p95 = calculate_percentile(values=ttfb_values, percentile=95.0)
        e2e_p95 = calculate_percentile(values=e2e_values, percentile=95.0)

        LOGGER.info(
            event="TTS latency results",
            request_count=len(ttfb_values),
            ttfb_p95_s=round(ttfb_p95, 4),
            ttfb_threshold_s=TTFB_P95_THRESHOLD_S,
            e2e_p95_s=round(e2e_p95, 4),
            e2e_threshold_s=E2E_P95_THRESHOLD_S,
        )

        assert ttfb_p95 <= TTFB_P95_THRESHOLD_S, (
            f"TTS latency failed: TTFB p95 {ttfb_p95:.3f}s exceeds threshold "
            f"{TTFB_P95_THRESHOLD_S}s (measured at OpenShift Route)"
        )
        assert e2e_p95 <= E2E_P95_THRESHOLD_S, (
            f"TTS latency failed: E2E p95 {e2e_p95:.3f}s exceeds threshold "
            f"{E2E_P95_THRESHOLD_S}s (measured at OpenShift Route)"
        )


@pytest.mark.vllm_omni_nvidia_single_gpu
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, vllm_omni_serving_runtime, vllm_omni_inference_service",
    [
        pytest.param(
            {"name": "vllm-omni-tts-soak"},
            {"model-dir": QWEN3_TTS_MODEL_PATH},
            {"deployment_mode": KServeDeploymentType.STANDARD},
            {
                "name": "vllm-omni-tts-soak",
                "model_path": QWEN3_TTS_MODEL_PATH,
                "deployment_mode": KServeDeploymentType.STANDARD,
                "gpu_count": 1,
            },
            id="test_vllm_omni_qwen3_tts_soak",
        ),
    ],
    indirect=True,
)
@pytest.mark.usefixtures("cluster_monitoring_config")
class TestVllmOmniSoakComposite:
    """Composite soak test (10-min continuous dialogue)."""

    def test_vllm_omni_composite_soak(
        self,
        vllm_omni_inference_service: InferenceService,
        vllm_omni_isvc_url: str,
        vllm_omni_pod_resource: Pod,
        prometheus: Prometheus | None,
    ) -> None:
        """Composite soak measures TTFB degradation, GPU memory growth, and stability.

        Given a vLLM-Omni ISVC with Qwen3-TTS on 1x NVIDIA GPU and Prometheus
        configured to scrape vllm:kv_cache_usage_perc,
        When OMNI_SOAK_DURATION seconds (default 600) of sequential /v1/audio/speech requests
        cycle through the fixed 50-prompt corpus after 2 warm-up requests,
        Then TTFB p95 at session end does not exceed 1.25x TTFB p95 at minute 1,
        GPU memory growth from baseline to session end is under 15%, and the pod
        restart count remains 0 throughout.
        """
        assert prometheus is not None, (
            "Composite soak requires Prometheus for GPU memory measurement. "
            "Ensure cluster monitoring is enabled and the prometheus fixture is configured."
        )

        raw_duration = os.environ.get("OMNI_SOAK_DURATION", "600")
        if not raw_duration.isdigit():
            pytest.fail(f"OMNI_SOAK_DURATION must be a positive integer, got {raw_duration!r}")
        soak_duration_s = int(raw_duration)
        if soak_duration_s < 180:
            pytest.fail(
                f"OMNI_SOAK_DURATION must be >= 180s so minute-1 and final measurement "
                f"windows (60s each) do not overlap, got {soak_duration_s}s"
            )
        base_url = vllm_omni_isvc_url

        warmup_tts(
            url=base_url,
            model=vllm_omni_inference_service.instance.metadata.name,
            voice=OMNI_TTS_VOICE[QWEN3_TTS_MODEL_PATH],
            count=WARM_UP_COUNT,
        )

        raw_baseline = None
        for attempt in range(4):
            raw_baseline = get_metrics_value(
                prometheus=prometheus,
                metrics_query=PROMETHEUS_KV_CACHE_METRIC,
            )
            if raw_baseline is not None:
                break
            LOGGER.debug(
                event="composite soak: waiting for GPU metric",
                attempt=attempt + 1,
                metric=PROMETHEUS_KV_CACHE_METRIC,
            )
            time.sleep(15)
        assert raw_baseline is not None, (
            f"Prometheus returned no data for '{PROMETHEUS_KV_CACHE_METRIC}' after 4 attempts (60s). "
            "Ensure cluster_monitoring_config fixture enabled user workload monitoring."
        )
        baseline_gpu_usage: float = float(raw_baseline)
        LOGGER.info(
            event="composite soak: baseline GPU kv-cache usage captured",
            baseline_gpu_usage=baseline_gpu_usage,
        )

        initial_restart_counts = get_restart_counts(pod=vllm_omni_pod_resource)
        prompt_cycle = itertools.cycle(TTS_PROMPT_CORPUS)  # noqa: FCN001
        start_time = time.monotonic()
        last_gpu_log_time = start_time

        min1_ttfb_values: list[float] = []
        final_window_ttfb_values: list[float] = []

        while (elapsed := time.monotonic() - start_time) < soak_duration_s:
            prompt = next(prompt_cycle)
            status_code, ttfb_s, _ = timed_tts_request(
                url=base_url,
                model=vllm_omni_inference_service.instance.metadata.name,
                prompt=prompt,
                voice=OMNI_TTS_VOICE[QWEN3_TTS_MODEL_PATH],
            )
            assert status_code == 200, f"Composite soak: request at T+{elapsed:.1f}s returned HTTP {status_code}"
            if time.monotonic() - last_gpu_log_time >= 120:
                gpu_snapshot = get_metrics_value(
                    prometheus=prometheus,
                    metrics_query=PROMETHEUS_KV_CACHE_METRIC,
                )
                LOGGER.info(
                    event="composite soak: GPU kv-cache snapshot",
                    elapsed_s=round(elapsed, 0),
                    kv_cache_usage=float(gpu_snapshot) if gpu_snapshot is not None else None,
                )
                last_gpu_log_time = time.monotonic()
            if elapsed <= 60.0:
                min1_ttfb_values.append(ttfb_s)
            if elapsed > soak_duration_s - 60:
                final_window_ttfb_values.append(ttfb_s)

        assert min1_ttfb_values, (
            f"Composite soak collected 0 TTFB samples during the first 60 seconds of "
            f"a {soak_duration_s}s soak. Increase OMNI_SOAK_DURATION or check request latency."
        )

        assert final_window_ttfb_values, (
            f"Composite soak collected 0 TTFB samples during the last 60 seconds of "
            f"a {soak_duration_s}s soak. Increase OMNI_SOAK_DURATION or check request latency."
        )

        ttfb_min1 = calculate_percentile(values=min1_ttfb_values, percentile=95.0)
        ttfb_final = calculate_percentile(values=final_window_ttfb_values, percentile=95.0)

        LOGGER.info(
            event="composite soak: TTFB measurements",
            ttfb_min1_p95_s=round(ttfb_min1, 4),
            ttfb_final_p95_s=round(ttfb_final, 4),
            degradation_ratio=round(ttfb_final / ttfb_min1, 3) if ttfb_min1 > 0 else None,
            threshold_ratio=SOAK_TTFB_DEGRADATION_RATIO,
        )

        ratio_text = f"{ttfb_final / ttfb_min1:.3f}" if ttfb_min1 > 0 else "n/a (minute-1 p95 was 0)"
        assert ttfb_final <= SOAK_TTFB_DEGRADATION_RATIO * ttfb_min1, (
            f"Composite soak failed: TTFB p95 at session end ({ttfb_final:.3f}s) exceeds "
            f"{SOAK_TTFB_DEGRADATION_RATIO}x the minute-1 baseline ({ttfb_min1:.3f}s). "
            f"Ratio: {ratio_text}"
        )

        assert_no_pod_restarts(
            pod=vllm_omni_pod_resource,
            initial_counts=initial_restart_counts,
            context="Composite soak",
        )

        raw_final_gpu = None
        for attempt in range(3):
            raw_final_gpu = get_metrics_value(
                prometheus=prometheus,
                metrics_query=PROMETHEUS_KV_CACHE_METRIC,
            )
            if raw_final_gpu is not None:
                break
            time.sleep(10)
        assert raw_final_gpu is not None, f"Prometheus returned no data for '{PROMETHEUS_KV_CACHE_METRIC}' after soak."
        final_gpu_usage = float(raw_final_gpu)
        if baseline_gpu_usage > 0:
            growth_ratio = (final_gpu_usage - baseline_gpu_usage) / baseline_gpu_usage
        else:
            assert final_gpu_usage <= SOAK_GPU_MEMORY_GROWTH_THRESHOLD, (
                f"Composite soak failed: baseline kv-cache usage was 0.0 and final usage is "
                f"{final_gpu_usage:.4f}, above the {SOAK_GPU_MEMORY_GROWTH_THRESHOLD:.0%} absolute limit."
            )
            growth_ratio = final_gpu_usage
        LOGGER.info(
            event="composite soak: GPU kv-cache usage",
            baseline=baseline_gpu_usage,
            final=final_gpu_usage,
            growth_ratio=round(growth_ratio, 4),
            threshold=SOAK_GPU_MEMORY_GROWTH_THRESHOLD,
        )
        assert growth_ratio < SOAK_GPU_MEMORY_GROWTH_THRESHOLD, (
            f"Composite soak failed: GPU kv-cache growth {growth_ratio:.1%} "
            f"exceeds {SOAK_GPU_MEMORY_GROWTH_THRESHOLD:.0%} threshold. "
            f"Baseline: {baseline_gpu_usage:.4f}, final: {final_gpu_usage:.4f}"
        )


@pytest.mark.vllm_omni_nvidia_single_gpu
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, vllm_omni_serving_runtime, vllm_omni_inference_service",
    [
        pytest.param(
            {"name": "vllm-omni-ov-stab"},
            {"model-dir": OMNIVOICE_MODEL_PATH},
            {"deployment_mode": KServeDeploymentType.STANDARD},
            {
                "name": "vllm-omni-ov-stab",
                "model_path": OMNIVOICE_MODEL_PATH,
                "deployment_mode": KServeDeploymentType.STANDARD,
                "gpu_count": 1,
            },
            id="test_vllm_omni_omnivoice",
        ),
        pytest.param(
            {"name": "vllm-omni-voxtral-stab"},
            {"model-dir": VOXTRAL_TTS_MODEL_PATH},
            {"deployment_mode": KServeDeploymentType.STANDARD},
            {
                "name": "vllm-omni-voxtral-stab",
                "model_path": VOXTRAL_TTS_MODEL_PATH,
                "deployment_mode": KServeDeploymentType.STANDARD,
                "gpu_count": 1,
            },
            id="test_vllm_omni_voxtral",
        ),
    ],
    indirect=True,
)
class TestVllmOmniVoiceModelStability:
    """50-turn session stability for voice models (OmniVoice and Voxtral-TTS).

    Parametrized across OmniVoice (k2-fsa, 0.6B) and Voxtral-TTS (4B).
    Each param deploys its own ISVC and validates 50 consecutive voice turns
    without pod restart or OOM events.
    """

    def test_vllm_omni_voice_50_turn_stability(
        self,
        vllm_omni_inference_service: InferenceService,
        vllm_omni_isvc_url: str,
        vllm_omni_pod_resource: Pod,
    ) -> None:
        """50 consecutive /v1/audio/speech turns without pod restart.

        Given a voice model InferenceService loaded from S3,
        When 50 sequential TTS requests are sent from the fixed prompt corpus,
        Then all return HTTP 200 and the pod restart count delta is 0.
        """
        model_name = vllm_omni_inference_service.instance.metadata.name
        storage_uri = vllm_omni_inference_service.instance.spec.predictor.model.storageUri or ""
        model_dir = storage_uri.rstrip("/").rsplit("/", 1)[-1]
        assert model_dir in OMNI_TTS_VOICE, (
            f"No voice mapping for model directory {model_dir!r} (storageUri {storage_uri!r}). "
            f"Add an OMNI_TTS_VOICE entry. Known keys: {sorted(OMNI_TTS_VOICE)}"
        )
        voice = OMNI_TTS_VOICE[model_dir]
        initial_restart_counts = get_restart_counts(pod=vllm_omni_pod_resource)
        LOGGER.info(
            event="voice model 50-turn stability: starting",
            model=model_name,
            voice=voice,
            pod=vllm_omni_pod_resource.name,
            initial_restart_counts=initial_restart_counts,
        )

        base_url = vllm_omni_isvc_url
        failed_turns: list[int] = []

        for turn_idx in range(FIFTY_TURN_COUNT):
            prompt = TTS_PROMPT_CORPUS[turn_idx % len(TTS_PROMPT_CORPUS)]
            status_code, _, _ = timed_tts_request(
                url=base_url,
                model=model_name,
                prompt=prompt,
                voice=voice,
            )
            log_loop_progress(
                logger=LOGGER,
                event="voice model 50-turn stability: turn result",
                current=turn_idx + 1,
                total=FIFTY_TURN_COUNT,
                is_error=status_code != 200,
                status_code=status_code,
            )
            if status_code != 200:
                failed_turns.append(turn_idx + 1)

        assert not failed_turns, (
            f"50-turn stability FAILED for {model_name!r}: "
            f"{len(failed_turns)}/50 turns returned non-200. Failed: {failed_turns}"
        )

        assert_no_pod_restarts(
            pod=vllm_omni_pod_resource,
            initial_counts=initial_restart_counts,
            context=f"50-turn stability for {model_name!r}",
        )
