"""Model+storage configurations — bind a model to a storage backend."""

from utilities.constants import ModelName, ModelStorage

from .config_base import CpuConfig, GpuConfig


class TinyLlamaOciConfig(CpuConfig):
    """TinyLlama via OCI container registry, CPU inference."""

    enable_auth = False
    name = "llmisvc-tinyllama-oci-cpu"
    storage_uri = ModelStorage.OCI.TINYLLAMA


class TinyLlamaS3Config(CpuConfig):
    """TinyLlama via S3 bucket, CPU inference."""

    enable_auth = False
    name = "llmisvc-tinyllama-s3-cpu"
    storage_uri = ModelStorage.S3.TINYLLAMA


class TinyLlamaHfConfig(CpuConfig):
    """TinyLlama via HuggingFace, CPU inference."""

    enable_auth = False
    name = "llmisvc-tinyllama-hf-cpu"
    storage_uri = ModelStorage.HuggingFace.TINYLLAMA
    wait_timeout = 420


class TinyLlamaOciGpuConfig(GpuConfig):
    """TinyLlama via OCI container registry, GPU inference."""

    enable_auth = False
    name = "llmisvc-tinyllama-oci-gpu"
    storage_uri = ModelStorage.OCI.TINYLLAMA
    model_name = ModelName.TINYLLAMA


class TinyLlamaS3GpuConfig(GpuConfig):
    """TinyLlama via S3 bucket, GPU inference."""

    enable_auth = False
    name = "llmisvc-tinyllama-s3-gpu"
    storage_uri = ModelStorage.S3.TINYLLAMA
    model_name = ModelName.TINYLLAMA


class TinyLlamaHfGpuConfig(GpuConfig):
    """TinyLlama via HuggingFace, GPU inference."""

    enable_auth = False
    name = "llmisvc-tinyllama-hf-gpu"
    storage_uri = ModelStorage.HuggingFace.TINYLLAMA
    model_name = ModelName.TINYLLAMA
