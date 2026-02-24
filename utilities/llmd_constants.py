"""LLMD-specific constants that extend the shared constants."""

from utilities.constants import (
    ContainerImages as SharedContainerImages,
)
from utilities.constants import (
    Labels,
    ModelName,
)
from utilities.constants import (
    ModelStorage as SharedModelStorage,
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
    DEFAULT_TIMEOUT: int = 60


class ModelStorage:
    """LLMD-specific model storage aliases for convenience."""

    TINYLLAMA_OCI: str = SharedModelStorage.OCI.TINYLLAMA
    TINYLLAMA_S3: str = SharedModelStorage.S3.TINYLLAMA
    S3_QWEN: str = SharedModelStorage.S3.QWEN_7B_INSTRUCT
    HF_TINYLLAMA: str = SharedModelStorage.HuggingFace.TINYLLAMA
    HF_OPT125M: str = SharedModelStorage.HuggingFace.OPT125M


class ContainerImages:
    """LLMD-specific container image aliases."""

    VLLM_CPU: str = SharedContainerImages.VLLM.CPU


class ModelNames:
    """LLMD-specific model name aliases."""

    QWEN: str = ModelName.QWEN
    TINYLLAMA: str = ModelName.TINYLLAMA


class LLMDDefaults:
    REPLICAS: int = 1
