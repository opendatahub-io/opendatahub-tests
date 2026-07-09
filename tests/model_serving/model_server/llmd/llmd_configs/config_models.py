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


class Qwen3MoeTinyRandomGpuConfig(GpuConfig):
    """Qwen3-MoE tiny random model via HuggingFace, GPU inference.

    ~10MB randomly initialized Qwen3-MoE with 8 experts. Produces garbage output
    but uses a real MoE architecture (qwen3_moe), so vLLM loads and serves it
    correctly. Useful for fast validation of MoE deployment plumbing without
    waiting for a real model to download.
    """

    enable_auth = False
    name = "llmisvc-qwen3-moe-tiny"
    storage_uri = "hf://yujiepan/qwen3-moe-tiny-random"
    model_name = "qwen3-moe-tiny-random"
