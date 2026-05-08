"""LLMD-specific constants that extend the shared constants."""

from utilities.constants import Labels


class LLMDGateway:
    DEFAULT_NAME: str = "openshift-ai-inference"
    DEFAULT_NAMESPACE: str = "openshift-ingress"
    DEFAULT_GATEWAY_CLASS: str = "openshift-default"


class KServeGateway:
    LABEL: str = Labels.Kserve.GATEWAY_LABEL
    INGRESS_GATEWAY: str = "kserve-ingress-gateway"
    API_GROUP: str = "gateway.networking.k8s.io"


class LLMEndpoint:
    CHAT_COMPLETIONS: str = "/v1/chat/completions"
    DEFAULT_MAX_TOKENS: int = 50
    DEFAULT_TEMPERATURE: float = 0.0
    DEFAULT_TIMEOUT: int = 60
