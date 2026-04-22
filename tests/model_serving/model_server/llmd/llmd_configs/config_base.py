"""Base configuration class for LLMInferenceService resources."""

from utilities.constants import ResourceLimits
from utilities.llmd_constants import ContainerImages


class LLMISvcConfig:
    """Base configuration for an LLMInferenceService resource.

    Subclass and override class attributes or classmethods for each test scenario.
    Pass the class directly to create_llmisvc_from_config — no instantiation needed.
    """

    name = ""
    model_name = None
    storage_uri = ""
    replicas = 1
    container_image = None
    template_config_ref = "kserve-config-llm-template"
    enable_auth = False
    wait_timeout = 240

    @classmethod
    def container_resources(cls):
        return {}

    @classmethod
    def container_env(cls):
        """Base environment variables for the vLLM container.

        Subclasses may either:
        - Call super().container_env() + [...] to extend the base env vars (used by CpuConfig)
        - Return a fresh list to fully replace (used by prefix cache configs that need
          exclusive control over VLLM_ADDITIONAL_ARGS)
        """
        return [
            {"name": "VLLM_LOGGING_LEVEL", "value": "DEBUG"},
        ]

    @classmethod
    def liveness_probe(cls):
        return {
            "httpGet": {"path": "/health", "port": 8000, "scheme": "HTTPS"},
            "initialDelaySeconds": 240,
            "periodSeconds": 60,
            "timeoutSeconds": 60,
            "failureThreshold": 10,
        }

    @classmethod
    def readiness_probe(cls):
        return None

    @classmethod
    def router_config(cls):
        return {
            "scheduler": {"configRef": "kserve-config-llm-scheduler"},
            "route": {},
            "gateway": {},
        }

    @classmethod
    def annotations(cls):
        return {
            "prometheus.io/port": "8000",
            "prometheus.io/path": "/metrics",
            "security.opendatahub.io/enable-auth": str(cls.enable_auth).lower(),
        }

    @classmethod
    def prefill_config(cls):
        return None

    @classmethod
    def labels(cls):
        return {}

    @classmethod
    def describe(cls, namespace: str = ""):
        """Return a formatted config summary for log output."""
        border = "=" * 60
        lines = [
            border,
            f"  Config: {cls.__name__}",
            border,
            f"  namespace:       {namespace}",
            f"  name:            {cls.name}",
            f"  storage_uri:     {cls.storage_uri}",
            f"  replicas:        {cls.replicas}",
            f"  container_image: {cls.container_image or '(default)'}",
            f"  auth:            {cls.annotations().get('security.opendatahub.io/enable-auth', 'false')}",
            border + "\n",
        ]
        return "\n".join(lines)

    @classmethod
    def with_overrides(cls, **overrides):
        """Create a derived config class with overridden attributes."""
        return type(f"{cls.__name__}_custom", (cls,), overrides)


class CpuConfig(LLMISvcConfig):
    """CPU inference base. Sets vLLM CPU image, CPU env vars, and CPU resource limits."""

    enable_auth = False
    container_image = ContainerImages.VLLM_CPU

    @classmethod
    def container_env(cls):
        # vLLM arguments to reduce engine startup time
        # --max-num-seqs 20
        # --max-model-len 128
        # --enforce-eager
        return super().container_env() + [
            {
                "name": "VLLM_ADDITIONAL_ARGS",
                "value": "--max-num-seqs 20 --max-model-len 128 --enforce-eager --ssl-ciphers ECDHE+AESGCM:DHE+AESGCM",
            },
            {"name": "VLLM_CPU_KVCACHE_SPACE", "value": "4"},
        ]

    @classmethod
    def container_resources(cls):
        return {
            "limits": {"cpu": "1", "memory": "10Gi"},
            "requests": {"cpu": "100m", "memory": "8Gi"},
        }


class GpuConfig(LLMISvcConfig):
    """GPU inference base. Sets GPU resource limits."""

    enable_auth = False
    wait_timeout = 600

    @classmethod
    def container_resources(cls):
        return {
            "limits": {
                "cpu": ResourceLimits.GPU.CPU_LIMIT,
                "memory": ResourceLimits.GPU.MEMORY_LIMIT,
                "nvidia.com/gpu": ResourceLimits.GPU.LIMIT,
            },
            "requests": {
                "cpu": ResourceLimits.GPU.CPU_REQUEST,
                "memory": ResourceLimits.GPU.MEMORY_REQUEST,
                "nvidia.com/gpu": ResourceLimits.GPU.REQUEST,
            },
        }
