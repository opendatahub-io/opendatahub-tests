"""LLMD-specific constants that extend the shared constants."""

from utilities.constants import (
    Timeout,
    ModelName,
    ContainerImages as SharedContainerImages,
    ModelStorage as SharedModelStorage,
    Labels,
)


class LLMDGateway:
    DEFAULT_NAME: str = "openshift-ai-inference"
    DEFAULT_NAMESPACE: str = "openshift-ingress"
    DEFAULT_CLASS: str = "data-science-gateway-class"


class KServeGateway:
    LABEL: str = Labels.Kserve.GATEWAY_LABEL
    INGRESS_GATEWAY: str = "kserve-ingress-gateway"
    API_GROUP: str = "gateway.networking.k8s.io"


class LLMEndpoint:
    CHAT_COMPLETIONS: str = "/v1/chat/completions"
    DEFAULT_MAX_TOKENS: int = 50
    DEFAULT_TEMPERATURE: float = 0.0
    DEFAULT_TIMEOUT: int = Timeout.TIMEOUT_30SEC


class ModelStorage:
    """LLMD-specific model storage aliases for convenience."""

    TINYLLAMA_OCI: str = SharedModelStorage.OCI.TINYLLAMA
    TINYLLAMA_S3: str = SharedModelStorage.S3.TINYLLAMA
    S3_QWEN: str = SharedModelStorage.S3.QWEN_7B_INSTRUCT
    HF_TINYLLAMA: str = SharedModelStorage.HuggingFace.TINYLLAMA


class ContainerImages:
    """LLMD-specific container image aliases."""

    VLLM_CPU: str = SharedContainerImages.VLLM.CPU


class ModelNames:
    """LLMD-specific model name aliases."""

    QWEN: str = ModelName.QWEN
    TINYLLAMA: str = ModelName.TINYLLAMA


class LLMDDefaults:
    REPLICAS: int = 1
