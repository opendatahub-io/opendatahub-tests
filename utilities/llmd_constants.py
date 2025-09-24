"""Centralized constants for LLMD (LLM Deployment) utilities and tests."""

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

VLLM_STORAGE_OCI = "oci://quay.io/mwaykole/test:opt-125m"
VLLM_CPU_IMAGE = "quay.io/pierdipi/vllm-cpu:latest"
DEFAULT_LLMD_REPLICAS = 1
DEFAULT_S3_STORAGE_PATH = "opt-125m"

DEFAULT_STORAGE_URI = VLLM_STORAGE_OCI
DEFAULT_CONTAINER_IMAGE = VLLM_CPU_IMAGE

DEFAULT_CPU_LIMIT = "1"
DEFAULT_MEMORY_LIMIT = "10Gi"
DEFAULT_CPU_REQUEST = "100m"
DEFAULT_MEMORY_REQUEST = "8Gi"

BASIC_LLMD_PARAMS = [({"name": "llmd-comprehensive-test"}, "openshift-default", "basic")]
