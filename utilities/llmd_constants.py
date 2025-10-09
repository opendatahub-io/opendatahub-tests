"""Centralized constants for LLMD (LLM Deployment) utilities and tests."""

import pytest
from utilities.constants import Timeout

DEFAULT_GATEWAY_NAME = "openshift-ai-inference"
DEFAULT_GATEWAY_NAMESPACE = "openshift-ingress"
OPENSHIFT_DEFAULT_GATEWAY_CLASS = "openshift-default"

KSERVE_GATEWAY_LABEL = "serving.kserve.io/gateway"
KSERVE_INGRESS_GATEWAY = "kserve-ingress-gateway"

DEFAULT_LLM_ENDPOINT = "/v1/chat/completions"
DEFAULT_MAX_TOKENS = 50
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TIMEOUT = Timeout.TIMEOUT_30SEC

VLLM_STORAGE_OCI = "oci://quay.io/mwaykole/test@sha256:f6691433a8fe554e60e42edcec4003aa0fec80f538d205530baf09840b3f36f1"
VLLM_STORAGE_HF = "hf://Qwen/Qwen2.5-7B-Instruct"
VLLM_CPU_IMAGE = "quay.io/pierdipi/vllm-cpu:latest"

DEFAULT_LLMD_REPLICAS = 1
DEFAULT_S3_STORAGE_PATH = "opt-125m"

DEFAULT_STORAGE_URI = VLLM_STORAGE_OCI
DEFAULT_CONTAINER_IMAGE = VLLM_CPU_IMAGE

DEFAULT_CPU_LIMIT = "1"
DEFAULT_MEMORY_LIMIT = "10Gi"
DEFAULT_CPU_REQUEST = "100m"
DEFAULT_MEMORY_REQUEST = "8Gi"

# GPU-specific constants
DEFAULT_GPU_LIMIT = "1"
DEFAULT_GPU_REQUEST = "1"
DEFAULT_GPU_CPU_LIMIT = "4"
DEFAULT_GPU_MEMORY_LIMIT = "32Gi"
DEFAULT_GPU_CPU_REQUEST = "2"
DEFAULT_GPU_MEMORY_REQUEST = "16Gi"

BASIC_LLMD_PARAMS = [({"name": "llmd-comprehensive-test"}, "basic")]

# GPU test parameters based on temp YAML configurations
GPU_LLMD_PARAMS = [
    # Standard GPU with scheduler (temp YAML: llm-inference-service-qwen2-7b-gpu.yaml)
    pytest.param({"name": "llmd-gpu-standard"}, {"name_suffix": "gpu-standard"}, id="gpu-standard"),
    # GPU without scheduler (temp YAML: llm-inference-service-qwen2-7b-gpu-no-scheduler.yaml)
    pytest.param(
        {"name": "llmd-gpu-no-scheduler"},
        {"name_suffix": "gpu-no-scheduler", "disable_scheduler": True},
        id="gpu-no-scheduler",
    ),
    # GPU with prefill/decode pattern (using smaller model for resource constraints)
    pytest.param(
        {"name": "llmd-gpu-pd"},
        {
            "name_suffix": "gpu-pd",
            "enable_prefill_decode": True,
            "replicas": 2,
            "prefill_replicas": 1,
            "storage_uri": "hf://TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        },
        id="gpu-prefill-decode",
    ),
]
