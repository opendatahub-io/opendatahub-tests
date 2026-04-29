"""Prefill-decode disaggregation configuration for LLMInferenceService."""

from utilities.constants import Labels

from .config_models import TinyLlamaS3GpuConfig


class PrefillDecodeConfig(TinyLlamaS3GpuConfig):
    """S3 GPU with prefill-decode disaggregation — inherits TinyLlama+S3+GPU."""

    enable_auth = False
    name = "llmisvc-prefill-decode-gpu"
    min_gpus_per_node = 2
    supported_accelerators = (Labels.Nvidia.NVIDIA_COM_GPU,)

    @classmethod
    def prefill_config(cls):
        return {
            "replicas": 1,
            "template": {
                "containers": [
                    {
                        "name": "main",
                        "resources": cls.container_resources(),
                        "env": [{"name": "VLLM_PREFILL_MODE", "value": "true"}],
                    }
                ],
            },
        }
