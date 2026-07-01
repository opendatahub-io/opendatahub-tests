"""KV cache CPU offloading configuration for single-node LLMInferenceService."""

from .config_models import TinyLlamaOciGpuConfig


class KvCacheCpuOffloadConfig(TinyLlamaOciGpuConfig):
    """Single-node GPU with KV cache CPU offloading via OffloadingConnector.

    Allocates 4 GiB of CPU RAM as the primary KV cache tier.
    The controller translates kvCacheOffloading.cpu into --kv-transfer-config
    with kv_connector=OffloadingConnector for vLLM.
    """

    name = "llmisvc-kv-cache-cpu-offload"
    kv_cache_cpu = "4Gi"

    @classmethod
    def kv_cache_offloading(cls) -> dict:
        """Return kvCacheOffloading spec: 4 GiB CPU tier with LRU eviction."""
        return {
            "cpu": cls.kv_cache_cpu,
            "evictionPolicy": "lru",
        }
