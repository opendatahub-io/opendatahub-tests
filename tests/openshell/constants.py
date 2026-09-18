import os

# Gateway connection — only needed when OPENSHELL_GATEWAY_URL is set (explicit override path)
OPENSHELL_GATEWAY_URL: str = os.getenv("OPENSHELL_GATEWAY_URL", "")
OPENSHELL_BEARER_TOKEN: str = os.getenv("OPENSHELL_BEARER_TOKEN", "")
OPENSHELL_TLS_CA_PATH: str = os.getenv("OPENSHELL_TLS_CA_PATH", "")
OPENSHELL_TLS_CERT_PATH: str = os.getenv("OPENSHELL_TLS_CERT_PATH", "")
OPENSHELL_TLS_KEY_PATH: str = os.getenv("OPENSHELL_TLS_KEY_PATH", "")

# vLLM inference provider
OPENSHELL_VLLM_ENDPOINT: str = os.getenv("OPENSHELL_VLLM_ENDPOINT", "")
OPENSHELL_VLLM_PROVIDER: str = os.getenv("OPENSHELL_VLLM_PROVIDER", "vllm")
OPENSHELL_VLLM_MODEL: str = os.getenv("OPENSHELL_VLLM_MODEL", "")
OPENSHELL_VLLM_TOKEN: str = os.getenv("OPENSHELL_VLLM_TOKEN", "fake")

# Sandbox
OPENSHELL_SANDBOX_OPENCODE_IMAGE: str = os.getenv("OPENSHELL_SANDBOX_OPENCODE_IMAGE", "")

# Helm-installed OpenShell PKI secrets
OPENSHELL_TLS_SERVER_SECRET_NAME: str = "openshell-server-tls"
OPENSHELL_TLS_CLIENT_SECRET_NAME: str = "openshell-client-tls"
OPENSHELL_NAMESPACE: str = "openshell"
