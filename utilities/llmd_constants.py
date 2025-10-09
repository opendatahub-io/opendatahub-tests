"""Centralized constants for LLMD (LLM Deployment) utilities and tests."""

from utilities.constants import Timeout


class LLMDGateway:
    DEFAULT_NAME: str = "openshift-ai-inference"
    DEFAULT_NAMESPACE: str = "openshift-ingress"
    DEFAULT_CLASS: str = "openshift-default"


class KServeGateway:
    LABEL: str = "serving.kserve.io/gateway"
    INGRESS_GATEWAY: str = "kserve-ingress-gateway"


class LLMEndpoint:
    CHAT_COMPLETIONS: str = "/v1/chat/completions"
    DEFAULT_MAX_TOKENS: int = 50
    DEFAULT_TEMPERATURE: float = 0.0
    DEFAULT_TIMEOUT: int = Timeout.TIMEOUT_30SEC


class ModelStorage:
    VLLM_OCI: str = (
        "oci://quay.io/mwaykole/test@sha256:f6691433a8fe554e60e42edcec4003aa0fec80f538d205530baf09840b3f36f1"
    )
    HF_QWEN: str = "hf://Qwen/Qwen2.5-7B-Instruct"
    HF_TINYLLAMA: str = "hf://TinyLlama/TinyLlama-1.1B-Chat-v1.0"


class ContainerImages:
    VLLM_CPU: str = "quay.io/pierdipi/vllm-cpu:latest"


class ModelNames:
    TINYLLAMA: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


class LLMDDefaults:
    REPLICAS: int = 1
    S3_STORAGE_PATH: str = "opt-125m"


class ResourceLimits:
    class CPU:
        LIMIT: str = "1"
        REQUEST: str = "100m"

    class Memory:
        LIMIT: str = "10Gi"
        REQUEST: str = "8Gi"

    class GPU:
        LIMIT: str = "1"
        REQUEST: str = "1"
        CPU_LIMIT: str = "4"
        CPU_REQUEST: str = "2"
        MEMORY_LIMIT: str = "32Gi"
        MEMORY_REQUEST: str = "16Gi"
