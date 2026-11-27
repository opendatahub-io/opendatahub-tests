"""Constants for vLLM-Omni serving runtime tests."""

import json
from pathlib import Path
from typing import Any

from tests.model_serving.model_runtime.vllm.constant import (
    PREDICT_RESOURCES,
)
from utilities.constants import AcceleratorType, Ports, RuntimeTemplates

# API endpoint paths
AUDIO_SPEECH_ENDPOINT: str = "/v1/audio/speech"
HEALTH_ENDPOINT: str = "/health"
IMAGES_GENERATIONS_ENDPOINT: str = "/v1/images/generations"

# Model mount path
MODEL_MOUNT_DIR: str = "/mnt/models"

# Prometheus metric constants
PROMETHEUS_KV_CACHE_METRIC: str = "vllm:kv_cache_usage_perc"
VLLM_METRIC_PREFIX: str = "vllm:"
VLLM_OMNI_METRIC_PREFIX: str = "vllm_omni:"

# Stability and performance constants
E2E_P95_THRESHOLD_S: float = 7.0
FIFTY_TURN_COUNT: int = 50
PERF_REQUEST_COUNT: int = 200
SOAK_GPU_MEMORY_GROWTH_THRESHOLD: float = 0.15
SOAK_TTFB_DEGRADATION_RATIO: float = 1.25
TTFB_P95_THRESHOLD_S: float = 3.5
WARM_UP_COUNT: int = 2

OMNI_TEMPLATE_MAP: dict[str, str] = {
    AcceleratorType.NVIDIA: RuntimeTemplates.VLLM_OMNI_CUDA,
}

# Resource specs — multi-GPU for Qwen3-Omni-30B, single-GPU for TTS models
OMNI_MULTI_GPU_RESOURCES: dict[str, Any] = {
    "resources": {
        "requests": {"cpu": "4", "memory": "32Gi"},
        "limits": {"cpu": "8", "memory": "64Gi"},
    },
}

OMNI_SINGLE_GPU_RESOURCES: dict[str, Any] = {
    "resources": {
        "requests": {"cpu": "2", "memory": "16Gi"},
        "limits": {"cpu": "4", "memory": "32Gi"},
    },
}

# Volumes/mounts shared with standard vLLM runtime
OMNI_VOLUMES: list[dict[str, Any]] = PREDICT_RESOURCES["volumes"]
OMNI_VOLUME_MOUNTS: list[dict[str, Any]] = PREDICT_RESOURCES["volume_mounts"]


# Probe configuration — matches vllm-omni-cuda-runtime-template spec exactly.
# Only fields explicitly set in the template are included; Kubernetes-defaulted
# fields (e.g. timeoutSeconds=1, successThreshold=1) are intentionally omitted.
# startup: max 1200 s window (40 x 30 s); no initialDelaySeconds on readiness/liveness
OMNI_STARTUP_PROBE: dict[str, Any] = {
    "httpGet": {"path": "/health", "port": 8080, "scheme": "HTTP"},
    "failureThreshold": 40,
    "periodSeconds": 30,
}

OMNI_READINESS_PROBE: dict[str, Any] = {
    "httpGet": {"path": "/health", "port": 8080, "scheme": "HTTP"},
    "periodSeconds": 10,
    "failureThreshold": 3,
}

OMNI_LIVENESS_PROBE: dict[str, Any] = {
    "httpGet": {"path": "/health", "port": 8080, "scheme": "HTTP"},
    "periodSeconds": 15,
    "failureThreshold": 3,
}

# Metrics endpoint configuration — matches ServingRuntime annotation
# prometheus.io/path: '/metrics', prometheus.io/port: '8080'
METRICS_PATH: str = "/metrics"
METRICS_PORT: int = Ports.REST_PORT  # 8080

# S3 model paths (relative paths under the models bucket)
QWEN3_OMNI_MODEL_PATH: str = "Qwen3-Omni-30B-A3B-Instruct"
QWEN3_TTS_MODEL_PATH: str = "Qwen3-TTS-12Hz-1.7B-CustomVoice"
VOXTRAL_TTS_MODEL_PATH: str = "Voxtral-4B-TTS-2603"
OMNIVOICE_MODEL_PATH: str = "OmniVoice-k2-fsa"
FLUX2_MODEL_PATH: str = "FLUX.2-klein-4B"

# Per-model voice preset for /v1/audio/speech requests
OMNI_TTS_VOICE: dict[str, str | None] = {
    QWEN3_TTS_MODEL_PATH: "vivian",
    VOXTRAL_TTS_MODEL_PATH: "casual_female",
    OMNIVOICE_MODEL_PATH: None,
}

# Base serving args — only used when a test explicitly needs to override template args.
# Most tests should NOT pass runtime_argument; the template args are used as-is.
OMNI_SERVING_ARGS: list[str] = [
    "--model=/mnt/models",
    "--omni",
    "--port=8080",
]

# TTS prompt corpus loaded from data/prompts.json
_PROMPTS_FILE = Path(__file__).parent / "data" / "prompts.json"
TTS_PROMPT_CORPUS: list[str] = (
    json.loads(_PROMPTS_FILE.read_text())
    if _PROMPTS_FILE.exists()
    else [
        "This is a default test prompt for text to speech validation.",
    ]
)
