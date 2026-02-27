"""Model+storage configurations — bind a model to a storage backend."""

from utilities.llmd_constants import ModelNames, ModelStorage

from .config_base import CpuConfig, GpuConfig


class TinyLlamaOciConfig(CpuConfig):
    """TinyLlama via OCI container registry, CPU inference."""

    enable_auth = False
    name = "llmisvc-tinyllama-oci-cpu"
    storage_uri = ModelStorage.TINYLLAMA_OCI


class TinyLlamaS3Config(CpuConfig):
    """TinyLlama via S3 bucket, CPU inference."""

    enable_auth = False
    name = "llmisvc-tinyllama-s3-cpu"
    storage_uri = ModelStorage.TINYLLAMA_S3


class Opt125mHfConfig(CpuConfig):
    """OPT-125M via HuggingFace, CPU inference."""

    enable_auth = False
    name = "llmisvc-opt125m-oci-cpu"
    storage_uri = ModelStorage.HF_OPT125M


class QwenS3Config(GpuConfig):
    """Qwen 7B via S3 bucket, GPU inference."""

    enable_auth = False
    name = "llmisvc-qwen-s3-gpu"
    storage_uri = ModelStorage.S3_QWEN
    model_name = ModelNames.QWEN


class QwenHfConfig(GpuConfig):
    """Qwen 7B via HuggingFace, GPU inference."""

    enable_auth = False
    name = "llmisvc-qwen-hf-gpu"
    storage_uri = ModelStorage.HF_QWEN_7B_INSTRUCT
    model_name = ModelNames.QWEN
