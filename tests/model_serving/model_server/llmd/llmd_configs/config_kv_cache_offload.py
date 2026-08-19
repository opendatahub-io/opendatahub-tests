"""KV cache offloading configurations — CPU and disk tier variants."""

from .config_models import TinyLlamaOciGpuConfig


class KvCacheCpuOffloadConfig(TinyLlamaOciGpuConfig):
    """TinyLlama via OCI, GPU inference with KV cache CPU offloading."""

    name = "llmisvc-kv-cache-cpu-offload"

    @classmethod
    def kv_cache_offloading(cls):
        return {"cpu": "4Gi", "evictionPolicy": "lru"}

    @classmethod
    def template_volumes(cls):
        # RHOAIENG-81261: kserve hardcodes /dev/shm at 1Gi but vLLM mmap needs ~4.3GB for CPU offload
        return [{"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "6Gi"}}]


class KvCacheDiskOffloadConfig(TinyLlamaOciGpuConfig):
    """TinyLlama via OCI, GPU inference with KV cache disk (filesystem) secondary tier."""

    name = "llmisvc-kv-cache-disk-offload"

    @classmethod
    def kv_cache_offloading(cls):
        return {
            "cpu": "4Gi",
            "secondary": [
                {"fileSystem": {"emptyDir": {"size": "20Gi"}}},
            ],
        }

    disk_volume_name = "kv-cache-secondary-0"
    disk_mount_path = "/mnt/kv-cache-0"
    container_name = "main"

    @classmethod
    def template_volumes(cls):
        # RHOAIENG-81261: kserve hardcodes /dev/shm at 1Gi but vLLM mmap needs ~4.3GB for CPU offload
        return [{"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "6Gi"}}]
