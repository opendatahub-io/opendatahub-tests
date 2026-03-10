"""Prefill-decode disaggregation configuration for LLMInferenceService."""

from .config_models import QwenS3Config


class PrefillDecodeConfig(QwenS3Config):
    """S3 GPU with prefill-decode disaggregation — inherits Qwen+S3+GPU from QwenS3Config."""

    enable_auth = False
    name = "llmisvc-prefill-decode-gpu"

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
