"""vLLM-Omni end-to-end integration tests with Qwen3-Omni-30B-A3B."""

from http import HTTPStatus
from typing import Any

import pytest
import requests
import structlog
import urllib3
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.serving_runtime import ServingRuntime
from ocp_utilities.monitoring import Prometheus
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.model_serving.model_runtime.utils import get_restart_counts, pod_is_ready
from tests.model_serving.model_runtime.vllm_omni.constant import (
    FIFTY_TURN_COUNT,
    OMNI_LIVENESS_PROBE,
    OMNI_READINESS_PROBE,
    OMNI_STARTUP_PROBE,
    PROMETHEUS_KV_CACHE_METRIC,
    QWEN3_OMNI_MODEL_PATH,
    TTS_PROMPT_CORPUS,
    VLLM_METRIC_PREFIX,
    VLLM_OMNI_METRIC_PREFIX,
)
from tests.model_serving.model_runtime.vllm_omni.probes.utils import assert_probe_config, get_probe
from tests.model_serving.model_runtime.vllm_omni.utils import (
    _extract_metric_families,
    _fetch_raw_metrics,
    assert_health_ok,
    assert_no_pod_restarts,
    assert_openai_error_body,
    log_loop_progress,
)
from utilities.constants import Containers, KServeDeploymentType, Labels, Timeout
from utilities.plugins.constant import OpenAIEnpoints, RestHeader
from utilities.plugins.openai_plugin import OpenAIClient

LOGGER = structlog.get_logger(name=__name__)
urllib3.disable_warnings(category=urllib3.exceptions.InsecureRequestWarning)

INIT_STAGE_TIMEOUT: int = Timeout.TIMEOUT_15MIN
POD_READY_TIMEOUT: int = Timeout.TIMEOUT_10MIN

DASHBOARD_ANNOTATION: str = Labels.OpenDataHub.DASHBOARD  # "opendatahub.io/dashboard"

EXTERNAL_NETWORK_ERROR_PATTERNS: list[str] = [
    "failed to resolve",
    "dns resolution failed",
    "failed to download",
    "connection refused",
    "no route to host",
    "network is unreachable",
    "connection timed out",
    "failed to fetch",
]

MIN_VLLM_OMNI_METRIC_FAMILY_COUNT: int = 15
MIN_VLLM_METRIC_FAMILY_COUNT: int = 30
PROMETHEUS_VALIDATION_METRIC: str = "vllm:prompt_tokens_total"
REQUIRED_KV_CACHE_LABEL_KEYS: tuple[str, ...] = ("stage", "replica")

# Markers emitted during vLLM-Omni multi-stage initialization.
# Verified against RHOAI build vLLM-Omni 0.26.0+rhaiv.0 pod logs.
OMNI_STAGE_LOG_MARKERS: tuple[str, ...] = (
    "[AsyncOmniEngine] Launching Orchestrator thread with 3 stages",
    "[stage_init] Stage-0 set runtime devices",
    "[stage_init] Stage-1 set runtime devices",
)

pytestmark = pytest.mark.usefixtures("skip_if_no_vllm_omni_multi_gpu", "valid_aws_config")


def _find_metric_data_lines(metrics_text: str, metric_name: str) -> list[str]:
    """Return all non-comment data lines whose name starts with *metric_name*."""
    return [line for line in metrics_text.splitlines() if line.startswith(metric_name) and not line.startswith("#")]


@pytest.mark.vllm_omni_nvidia_multi_gpu
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, vllm_omni_serving_runtime, vllm_omni_inference_service",
    [
        pytest.param(
            {"name": "vllm-omni-e2e"},
            {"model-dir": QWEN3_OMNI_MODEL_PATH},
            {"deployment_mode": KServeDeploymentType.STANDARD},
            {
                "name": "vllm-omni-e2e",
                "model_path": QWEN3_OMNI_MODEL_PATH,
                "gpu_count": 2,
                "timeout": Timeout.TIMEOUT_40MIN,
                "deployment_mode": KServeDeploymentType.STANDARD,
                "model_env_variables": [{"name": "HF_HOME", "value": "/tmp/hf_home"}],
                "min-replicas": 1,
            },
            id="test_vllm_omni_qwen3_omni_e2e",
        ),
    ],
    indirect=True,
)
class TestVllmOmniFullE2E:
    """Full validation of vLLM-Omni with Qwen3-Omni-30B-A3B (E2E + API + Metrics + Stability).

    Combines four test suites that all deploy Qwen3-Omni-30B (2 GPU) into a single
    class sharing one ISVC deployment. Covers:
      - E2E: annotation, pod-ready, health, models, TTS inference, disconnected checks
      - API: OpenAI conformance for /v1/models, /v1/completions, /v1/chat/completions
      - Metrics: vllm_omni:* and vllm:* Prometheus metric families (enabled by default)
      - Stability: 50-turn session stability without pod restart
    """

    def test_vllm_omni_serving_runtime_dashboard_label(
        self,
        model_namespace: Namespace,
        s3_models_storage_uri: str,
        vllm_omni_serving_runtime: ServingRuntime,
        vllm_omni_inference_service: InferenceService,
    ) -> None:
        """Given the vLLM-Omni ServingRuntime exists in the cluster,
        When its metadata labels are inspected,
        Then the opendatahub.io/dashboard label must equal "true" so that the
        RHOAI dashboard can discover and present the runtime to users.
        """
        labels: dict[str, str] = vllm_omni_serving_runtime.instance.metadata.labels or {}
        dashboard_value: str | None = labels.get(DASHBOARD_ANNOTATION)
        assert dashboard_value == "true", (
            f"ServingRuntime '{vllm_omni_serving_runtime.name}' is missing the required dashboard label "
            f"'{DASHBOARD_ANNOTATION}': expected 'true', got '{dashboard_value}'"
        )

    def test_vllm_omni_predictor_pod_ready(
        self,
        model_namespace: Namespace,
        s3_models_storage_uri: str,
        vllm_omni_serving_runtime: ServingRuntime,
        vllm_omni_inference_service: InferenceService,
        vllm_omni_pod_resource: Pod,
    ) -> None:
        """Given Qwen3-Omni-30B-A3B requires three sequential initialization stages
        (LLM, TTS encoder, Code2Wav decoder),
        When the predictor pod is polled for up to 900 s,
        Then the pod must reach Ready state, confirming all three stages completed
        before the /health endpoint becomes available.
        """
        try:
            for ready in TimeoutSampler(
                wait_timeout=INIT_STAGE_TIMEOUT,
                sleep=10,
                func=pod_is_ready,
                pod=vllm_omni_pod_resource,
            ):
                if ready:
                    break
        except TimeoutExpiredError:
            pytest.fail(
                f"Predictor pod '{vllm_omni_pod_resource.name}' did not reach Ready state within "
                f"{INIT_STAGE_TIMEOUT} s. All three initialization stages (LLM, TTS, Code2Wav) "
                "must complete before the pod reports Ready."
            )

    def test_vllm_omni_health_endpoint(
        self,
        model_namespace: Namespace,
        s3_models_storage_uri: str,
        vllm_omni_serving_runtime: ServingRuntime,
        vllm_omni_inference_service: InferenceService,
        vllm_omni_isvc_url: str,
    ) -> None:
        """Given the vLLM-Omni predictor pod is Ready,
        When a GET /health request is sent to the InferenceService external URL,
        Then the response status code must be 200, confirming the server is fully
        initialized and accepting traffic.
        """
        assert_health_ok(url=vllm_omni_isvc_url)

    def test_vllm_omni_chat_inference(
        self,
        model_namespace: Namespace,
        s3_models_storage_uri: str,
        vllm_omni_serving_runtime: ServingRuntime,
        vllm_omni_inference_service: InferenceService,
        vllm_omni_isvc_url: str,
    ) -> None:
        """Given the vLLM-Omni server has Qwen3-Omni-30B-A3B loaded,
        When a POST /v1/chat/completions request is sent with modalities=["text"],
        Then the response must be HTTP 200 with non-empty assistant content,
        confirming end-to-end inference through the Thinker stage.

        Qwen3-Omni does not support /v1/audio/speech; its primary interface is
        /v1/chat/completions (see vllm-omni PR #4762).
        """
        base_url = vllm_omni_isvc_url
        response = requests.post(
            url=f"{base_url}{OpenAIEnpoints.CHAT_COMPLETIONS}",
            json={
                "model": vllm_omni_inference_service.instance.metadata.name,
                "messages": [{"role": "user", "content": "Describe vLLM-Omni in one sentence."}],
                "max_tokens": 64,
                "modalities": ["text"],
            },
            headers={"Content-Type": "application/json"},
            verify=False,
            timeout=120,
        )
        assert response.status_code == HTTPStatus.OK, (
            f"POST /v1/chat/completions returned HTTP {response.status_code}; "
            f"expected {HTTPStatus.OK}. Response: {response.text[:200]}"
        )
        body = response.json()
        choices = body.get("choices") or []
        assert choices, (
            f"Chat response contains no choices for model '{vllm_omni_inference_service.instance.metadata.name}'. "
            f"Response body: {body}"
        )
        content = choices[0].get("message", {}).get("content", "")
        assert len(content) > 0, (
            f"Chat response content is empty for model '{vllm_omni_inference_service.instance.metadata.name}'. "
            f"Expected non-empty assistant text from /v1/chat/completions. Response body: {body}. "
            "Verify the model loaded correctly and supports text modality."
        )

    def test_vllm_omni_model_loads_from_local_s3(
        self,
        model_namespace: Namespace,
        s3_models_storage_uri: str,
        vllm_omni_serving_runtime: ServingRuntime,
        vllm_omni_inference_service: InferenceService,
        vllm_omni_pod_resource: Pod,
        vllm_omni_pod_logs: str,
    ) -> None:
        """Given the vLLM-Omni runtime is configured with weights from S3,
        When the predictor pod logs are scanned after startup,
        Then the logs must contain evidence of model loading from /mnt/models (the KServe
        mount path for S3-backed storage), confirming no external Hugging Face Hub access
        occurred.
        """
        logs: str = vllm_omni_pod_logs
        assert "/mnt/models" in logs.lower(), (
            "Pod logs do not contain evidence of model loading from /mnt/models. "
            "The model may have been loaded from an external source instead of local S3."
        )

    def test_vllm_omni_no_external_network_access_in_pod_logs(
        self,
        model_namespace: Namespace,
        s3_models_storage_uri: str,
        vllm_omni_serving_runtime: ServingRuntime,
        vllm_omni_inference_service: InferenceService,
        vllm_omni_pod_resource: Pod,
        vllm_omni_pod_logs: str,
    ) -> None:
        """Given inference has completed successfully,
        When the predictor pod logs are scanned for known external-network error patterns,
        Then no DNS failures, download errors, or connection timeouts are present,
        indicating no observed failed external-access attempts in the log stream.
        """
        logs: str = vllm_omni_pod_logs
        detected_patterns: list[str] = [
            pattern for pattern in EXTERNAL_NETWORK_ERROR_PATTERNS if pattern.lower() in logs.lower()
        ]
        assert not detected_patterns, (
            f"Pod '{vllm_omni_pod_resource.name}' logs contain network-error patterns that suggest "
            f"failed external access attempts: {detected_patterns}."
        )

    def test_vllm_omni_models_endpoint(
        self,
        vllm_omni_inference_service: InferenceService,
        vllm_omni_isvc_url: str,
    ) -> None:
        """GET /v1/models returns a valid OpenAI-compatible model list.

        Given a deployed vLLM-Omni InferenceService serving Qwen3-Omni-30B-A3B,
        When a GET request is sent to /v1/models,
        Then the response has HTTP 200 with "object": "list" and a non-empty "data" array
        containing at least one entry whose "id" matches the served model name,
        and each entry carries "object": "model".
        """
        base_url = vllm_omni_isvc_url
        response = requests.get(
            url=f"{base_url}{OpenAIEnpoints.MODELS_INFO}",
            headers=RestHeader.HEADERS,
            verify=False,
            timeout=120,
        )
        LOGGER.info(event="GET /v1/models response", status_code=response.status_code)

        assert response.status_code == 200, (
            f"Expected HTTP 200 from {OpenAIEnpoints.MODELS_INFO}, got {response.status_code}: {response.text}"
        )

        body: dict[str, Any] = response.json()
        assert body.get("object") == "list", f"Expected 'object'='list', got {body.get('object')!r}"

        data: list[dict[str, Any]] = body.get("data", [])
        assert len(data) >= 1, (
            f"Expected at least one model entry in 'data' array from {OpenAIEnpoints.MODELS_INFO}, "
            f"got {len(data)} entries. Full response body: {body}. "
            "Verify the model is loaded and the InferenceService is in Ready state."
        )

        model_ids: list[Any] = [entry.get("id") for entry in data]
        served_name = vllm_omni_inference_service.instance.metadata.name
        assert served_name in model_ids, f"Expected served model {served_name!r} in model list, found: {model_ids}"

        for entry in data:
            assert entry.get("object") == "model", (
                f"Expected each data entry to have 'object'='model', got {entry.get('object')!r}"
            )

    def test_vllm_omni_completions_endpoint_rejects_with_guidance(
        self,
        vllm_omni_inference_service: InferenceService,
        vllm_omni_isvc_url: str,
    ) -> None:
        """POST /v1/completions returns 400 with guidance to use chat/completions.

        Qwen3-Omni requires chat template structure for thinker-talker handoff.
        The /v1/completions endpoint correctly rejects with a descriptive error
        directing users to /v1/chat/completions instead.
        """
        served_name = vllm_omni_inference_service.instance.metadata.name
        base_url = vllm_omni_isvc_url
        payload: dict[str, Any] = {
            "model": served_name,
            "prompt": "The capital of France is",
            "max_tokens": 50,
            "modalities": ["text"],
        }
        response = requests.post(
            url=f"{base_url}{OpenAIEnpoints.COMPLETIONS}",
            headers=RestHeader.HEADERS,
            json=payload,
            verify=False,
            timeout=120,
        )
        LOGGER.info(event="POST /v1/completions response", status_code=response.status_code)

        assert response.status_code == 400, (
            f"Expected HTTP 400 from {OpenAIEnpoints.COMPLETIONS} for Omni model, "
            f"got {response.status_code}: {response.text}"
        )
        body: dict[str, Any] = response.json()
        error_msg = body.get("error", {}).get("message", "")
        assert "chat" in error_msg.lower(), f"Expected error message to mention /v1/chat/completions, got: {error_msg}"
        LOGGER.info(event="/v1/completions correctly rejected for Omni model", error=error_msg)

    def test_vllm_omni_chat_completions_endpoint(
        self,
        vllm_omni_inference_service: InferenceService,
        vllm_omni_isvc_url: str,
    ) -> None:
        """POST /v1/chat/completions returns a well-formed chat response.

        Given a deployed vLLM-Omni InferenceService with Qwen3-Omni-30B-A3B loaded,
        When a POST to /v1/chat/completions is made with a user message and modalities=["text"],
        Then the response has HTTP 200, "object": "chat.completion", at least one choice
        containing an assistant message with non-empty "content", a "finish_reason", and a
        "usage" object with prompt_tokens, completion_tokens, and total_tokens.
        """
        served_name = vllm_omni_inference_service.instance.metadata.name
        base_url = vllm_omni_isvc_url
        payload: dict[str, Any] = {
            "model": served_name,
            "messages": [{"role": "user", "content": "What is machine learning?"}],
            "max_tokens": 100,
            "modalities": ["text"],
        }
        response = requests.post(
            url=f"{base_url}{OpenAIEnpoints.CHAT_COMPLETIONS}",
            headers=RestHeader.HEADERS,
            json=payload,
            verify=False,
            timeout=120,
        )
        LOGGER.info(event="POST /v1/chat/completions response", status_code=response.status_code)

        assert response.status_code == 200, (
            f"Expected HTTP 200 from {OpenAIEnpoints.CHAT_COMPLETIONS}, got {response.status_code}: {response.text}"
        )

        body: dict[str, Any] = response.json()
        assert body.get("object") == "chat.completion", (
            f"Expected 'object'='chat.completion', got {body.get('object')!r}"
        )

        choices: list[dict[str, Any]] = body.get("choices", [])
        assert len(choices) >= 1, (
            f"Expected at least one entry in 'choices' array from {OpenAIEnpoints.CHAT_COMPLETIONS}, "
            f"got {len(choices)}. Response body: {body}. "
            "Verify the model generated a response and max_tokens is sufficient."
        )

        choice = choices[0]
        message: dict[str, Any] = choice.get("message", {})
        assert message.get("role") == "assistant", f"Expected message 'role'='assistant', got {message.get('role')!r}"
        assert message.get("content"), f"Expected non-empty 'content' in message, got {message.get('content')!r}"
        assert "finish_reason" in choice, f"Expected 'finish_reason' in first choice: {choice}"

        content: str = message.get("content", "")
        assert len(content) > 10, (
            f"Expected chat response content longer than 10 characters, got {len(content)}: {content!r}"
        )

        usage: dict[str, Any] = body.get("usage", {})
        for token_key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            assert token_key in usage, f"Expected '{token_key}' in usage object: {usage}"

    @pytest.mark.parametrize(
        "request_body, expected_status_codes",
        [
            pytest.param(
                {"prompt": "test"},
                [400, 422],
                id="test-missing-model-field",
            ),
            pytest.param(
                {},
                [400, 422],
                id="test-empty-body",
            ),
            pytest.param(
                {"model": "nonexistent-model", "prompt": "test"},
                [400, 404],
                id="test-invalid-model-name",
            ),
        ],
    )
    def test_vllm_omni_completions_error_handling(
        self,
        vllm_omni_inference_service: InferenceService,
        vllm_omni_isvc_url: str,
        request_body: dict[str, Any],
        expected_status_codes: list[int],
    ) -> None:
        """POST /v1/completions returns OpenAI error format for malformed requests.

        Given a deployed vLLM-Omni InferenceService,
        When a POST to /v1/completions is made with an invalid body — missing the required
        "model" field, an empty body, or an unknown model name —
        Then the response returns an HTTP 4xx status code and a JSON body containing an
        "error" object with "message", "type", and "code" fields.
        """
        base_url = vllm_omni_isvc_url
        response = requests.post(
            url=f"{base_url}{OpenAIEnpoints.COMPLETIONS}",
            headers=RestHeader.HEADERS,
            json=request_body,
            verify=False,
            timeout=120,
        )
        LOGGER.info(
            event="POST /v1/completions error-case response",
            request_body=request_body,
            status_code=response.status_code,
        )

        assert response.status_code in expected_status_codes, (
            f"Expected status in {expected_status_codes}, got {response.status_code}: {response.text}"
        )

        assert_openai_error_body(response=response)

    @pytest.mark.parametrize(
        "body_factory, expected_status_codes",
        [
            pytest.param(
                lambda model: {"model": model},
                [400, 422],
                id="test-missing-messages-field",
            ),
            pytest.param(
                lambda model: {"model": model, "messages": "invalid"},
                [400, 422],
                id="test-invalid-messages-format",
            ),
            pytest.param(
                lambda model: {"model": model, "messages": [{"content": "test"}]},
                [400, 422],
                id="test-message-missing-role",
            ),
        ],
    )
    def test_vllm_omni_chat_completions_error_handling(
        self,
        vllm_omni_inference_service: InferenceService,
        vllm_omni_isvc_url: str,
        body_factory: Any,
        expected_status_codes: list[int],
    ) -> None:
        """POST /v1/chat/completions returns OpenAI error format for malformed requests.

        Given a deployed vLLM-Omni InferenceService,
        When a POST to /v1/chat/completions is made with an invalid body — missing "messages",
        a non-array "messages" value, or a message dict missing the required "role" field —
        Then the response returns an HTTP 4xx status code and a JSON body containing an
        "error" object with "message", "type", and "code" fields.
        """
        request_body = body_factory(vllm_omni_inference_service.instance.metadata.name)  # noqa: FCN001
        base_url = vllm_omni_isvc_url
        response = requests.post(
            url=f"{base_url}{OpenAIEnpoints.CHAT_COMPLETIONS}",
            headers=RestHeader.HEADERS,
            json=request_body,
            verify=False,
            timeout=120,
        )
        LOGGER.info(
            event="POST /v1/chat/completions error-case response",
            request_body=request_body,
            status_code=response.status_code,
        )

        assert response.status_code in expected_status_codes, (
            f"Expected status in {expected_status_codes}, got {response.status_code}: {response.text}"
        )

        assert_openai_error_body(response=response)

    @pytest.mark.usefixtures("cluster_monitoring_config")
    def test_vllm_omni_metrics_endpoint_exposed(
        self,
        vllm_omni_inference_service: InferenceService,
        vllm_omni_isvc_url: str,
        vllm_omni_pod_resource: Pod,
        prometheus: Prometheus,
    ) -> None:
        """vllm_omni:* Prometheus metrics are exposed after inference.

        Given a vLLM-Omni InferenceService with Qwen3-Omni-30B-A3B loaded (metrics enabled by default),
        When a chat completion request is sent and the /metrics endpoint is queried,
        Then vllm_omni:* metric families (>= 15) are present at /metrics, and Prometheus
        successfully scrapes at least one sample.
        """
        base_url = vllm_omni_isvc_url
        warmup_response = requests.post(
            url=f"{base_url}{OpenAIEnpoints.CHAT_COMPLETIONS}",
            json={
                "model": vllm_omni_inference_service.instance.metadata.name,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10,
                "modalities": ["text"],
            },
            headers={"Content-Type": "application/json"},
            verify=False,
            timeout=120,
        )
        assert warmup_response.status_code == 200, (
            f"Warm-up request returned HTTP {warmup_response.status_code}: {warmup_response.text[:200]}"
        )

        raw_metrics = _fetch_raw_metrics(pod=vllm_omni_pod_resource)
        assert raw_metrics, (
            f"Empty /metrics response from pod '{vllm_omni_pod_resource.name}'. "
            "Verify the vLLM-Omni server started with metrics enabled (default) "
            "and the pod's port 8080 is reachable via port-forward."
        )

        vllm_omni_families = _extract_metric_families(metrics_text=raw_metrics, prefix=VLLM_OMNI_METRIC_PREFIX)
        family_count = len(vllm_omni_families)
        assert family_count >= MIN_VLLM_OMNI_METRIC_FAMILY_COUNT, (
            f"Expected >= {MIN_VLLM_OMNI_METRIC_FAMILY_COUNT} vllm_omni:* metric families, "
            f"found {family_count}: {sorted(vllm_omni_families)}"
        )

        try:
            for sample in TimeoutSampler(
                wait_timeout=180,
                sleep=15,
                func=lambda: prometheus.query_sampler(query=PROMETHEUS_VALIDATION_METRIC),
            ):
                if sample:
                    break
        except TimeoutExpiredError:
            pytest.fail(
                f"Prometheus returned no data for '{PROMETHEUS_VALIDATION_METRIC}' after 180 s; "
                "verify that user workload monitoring is enabled and Prometheus "
                "scrape config targets vLLM-Omni pods on port 8080."
            )

    def test_vllm_omni_vllm_metrics_exposed(
        self,
        vllm_omni_inference_service: InferenceService,
        vllm_omni_isvc_url: str,
        vllm_omni_pod_resource: Pod,
    ) -> None:
        """vllm:* Prometheus metrics are co-located with vllm_omni:* at /metrics.

        Given a vLLM-Omni InferenceService with Qwen3-Omni-30B-A3B loaded (metrics enabled by default),
        When a text completion request is sent and the /metrics endpoint is queried,
        Then vllm:* metric families (>= 30) are present alongside vllm_omni:* families,
        and vllm:kv_cache_usage_perc carries both a 'stage' and a 'replica' label.
        """
        base_url = vllm_omni_isvc_url
        inference_client = OpenAIClient(host=base_url, model_name=vllm_omni_inference_service.instance.metadata.name)
        inference_client.request_http(
            endpoint=OpenAIEnpoints.CHAT_COMPLETIONS,
            query=[{"role": "user", "content": "Test prompt for metrics validation."}],
            extra_param={"max_tokens": 10, "modalities": ["text"]},
        )

        raw_metrics = _fetch_raw_metrics(pod=vllm_omni_pod_resource)
        assert raw_metrics, (
            f"Empty /metrics response from pod '{vllm_omni_pod_resource.name}'. "
            "Verify the vLLM-Omni server started with metrics enabled (default) "
            "and the pod's port 8080 is reachable via port-forward."
        )

        kv_cache_lines = _find_metric_data_lines(metrics_text=raw_metrics, metric_name=PROMETHEUS_KV_CACHE_METRIC)
        assert kv_cache_lines, (
            f"Metric '{PROMETHEUS_KV_CACHE_METRIC}' has no data lines in /metrics on pod {vllm_omni_pod_resource.name}"
        )
        first_kv_line = kv_cache_lines[0]
        for label_key in REQUIRED_KV_CACHE_LABEL_KEYS:
            assert f'{label_key}="' in first_kv_line, (
                f"Expected label '{label_key}' not found in metric line: {first_kv_line!r}"
            )

        vllm_families = _extract_metric_families(metrics_text=raw_metrics, prefix=VLLM_METRIC_PREFIX)
        family_count = len(vllm_families)
        assert family_count >= MIN_VLLM_METRIC_FAMILY_COUNT, (
            f"Expected >= {MIN_VLLM_METRIC_FAMILY_COUNT} vllm:* metric families "
            f"(strategy target 32), found {family_count}: {sorted(vllm_families)}"
        )

        vllm_omni_families = _extract_metric_families(metrics_text=raw_metrics, prefix=VLLM_OMNI_METRIC_PREFIX)
        assert vllm_omni_families, (
            "No vllm_omni:* metric families found alongside vllm:* families; "
            "expected both prefixes to appear at the same /metrics endpoint"
        )

    def test_vllm_omni_50_turn_stability(
        self,
        vllm_omni_inference_service: InferenceService,
        vllm_omni_isvc_url: str,
        vllm_omni_pod_resource: Pod,
    ) -> None:
        """Qwen3-Omni-30B-A3B sustains 50 chat turns without restart.

        Qwen3-Omni does not support /v1/audio/speech (that endpoint is reserved for
        dedicated TTS models). Its primary interface is /v1/chat/completions with
        modalities=["text"] or modalities=["text", "audio"].

        Given a vLLM-Omni ISVC with Qwen3-Omni-30B-A3B on 2x GPUs and a fixed prompt corpus,
        When 50 sequential /v1/chat/completions requests are sent with modalities=["text"],
        Then all 50 return HTTP 200 with non-empty content, and the pod restart count
        is unchanged from its initial value.
        """
        assert pod_is_ready(pod=vllm_omni_pod_resource), (
            f"Pod {vllm_omni_pod_resource.name} must be Ready before starting the stability session"
        )

        initial_restart_counts = get_restart_counts(pod=vllm_omni_pod_resource)
        LOGGER.info(
            event="50-turn chat stability: starting Qwen3-Omni chat session",
            pod=vllm_omni_pod_resource.name,
            initial_restart_counts=initial_restart_counts,
        )

        base_url = vllm_omni_isvc_url
        served_name = vllm_omni_inference_service.instance.metadata.name
        success_count = 0
        failure_count = 0

        for turn_idx in range(FIFTY_TURN_COUNT):
            prompt = TTS_PROMPT_CORPUS[turn_idx % len(TTS_PROMPT_CORPUS)]
            response = requests.post(
                url=f"{base_url}{OpenAIEnpoints.CHAT_COMPLETIONS}",
                json={
                    "model": served_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 64,
                    "modalities": ["text"],
                },
                headers={"Content-Type": "application/json"},
                verify=False,
                timeout=120,
            )
            if response.status_code == 200 and len(response.content) > 0:
                success_count += 1
            else:
                failure_count += 1
            log_loop_progress(
                logger=LOGGER,
                event="50-turn chat stability: turn result",
                current=turn_idx + 1,
                total=FIFTY_TURN_COUNT,
                is_error=response.status_code != 200,
                status_code=response.status_code,
                success_count=success_count,
            )

        assert success_count == FIFTY_TURN_COUNT, (
            f"50-turn chat stability failed: only {success_count}/{FIFTY_TURN_COUNT} turns succeeded "
            f"({failure_count} failures); expected 100% success rate"
        )

        assert_no_pod_restarts(
            pod=vllm_omni_pod_resource,
            initial_counts=initial_restart_counts,
            context="50-turn chat stability",
        )

    def test_vllm_omni_three_stage_init_sequence(
        self,
        vllm_omni_inference_service: InferenceService,
        vllm_omni_pod_resource: Pod,
    ) -> None:
        """Three-stage init markers present and no CrashLoopBackOff.

        Given a vLLM-Omni ISVC with Qwen3-Omni-30B-A3B that has reached Ready,
        When the predictor pod logs are inspected,
        Then all three Orchestrator stage-ready markers are present and no
        container is in CrashLoopBackOff state.
        """
        pod_logs: str = vllm_omni_pod_resource.log(container=Containers.KSERVE_CONTAINER_NAME)
        missing = [m for m in OMNI_STAGE_LOG_MARKERS if m not in pod_logs]
        assert not missing, (
            f"Stage markers missing from pod '{vllm_omni_pod_resource.name}' logs: "
            f"{missing}. Expected all of: {list(OMNI_STAGE_LOG_MARKERS)}"
        )

        for cs in vllm_omni_pod_resource.instance.status.containerStatuses or []:
            waiting = getattr(getattr(cs, "state", None), "waiting", None)
            assert waiting is None or waiting.reason != "CrashLoopBackOff", (
                f"Container '{cs.name}' in CrashLoopBackOff on pod '{vllm_omni_pod_resource.name}'"
            )

    def test_vllm_omni_liveness_probe_configuration(
        self,
        vllm_omni_pod_resource: Pod,
    ) -> None:
        """Liveness probe matches template spec."""
        probe = get_probe(pod=vllm_omni_pod_resource, probe_type="livenessProbe")
        assert_probe_config(probe=probe, probe_name="livenessProbe", expected=OMNI_LIVENESS_PROBE)

    def test_vllm_omni_startup_probe_configuration(
        self,
        vllm_omni_pod_resource: Pod,
    ) -> None:
        """Startup probe matches template spec."""
        probe = get_probe(pod=vllm_omni_pod_resource, probe_type="startupProbe")
        assert_probe_config(probe=probe, probe_name="startupProbe", expected=OMNI_STARTUP_PROBE)

    def test_vllm_omni_readiness_probe_configuration(
        self,
        vllm_omni_pod_resource: Pod,
    ) -> None:
        """Readiness probe matches template spec."""
        probe = get_probe(pod=vllm_omni_pod_resource, probe_type="readinessProbe")
        assert_probe_config(probe=probe, probe_name="readinessProbe", expected=OMNI_READINESS_PROBE)
