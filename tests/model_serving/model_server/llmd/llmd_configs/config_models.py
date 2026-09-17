"""Model+storage configurations — bind a model to a storage backend."""

from tests.model_serving.model_server.llmd.constants import LLMISVC_S3_CONNECTION_PATH
from tests.model_serving.model_server.llmd.utils import (
    assert_llmisvc_oci_injected,
    assert_llmisvc_s3_fully_injected,
    assert_llmisvc_uri_injected,
)
from utilities.constants import ModelName, ModelStorage
from utilities.resources.llm_inference_service import LLMInferenceService

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


class TinyLlamaS3ConnectionConfig(CpuConfig):
    """TinyLlama via an S3 ConnectionsAPI connection Secret, CPU inference.

    `storage_uri` is a placeholder: the odh-model-controller ConnectionsAPI webhook overwrites
    `spec.model.uri` from the referenced connection Secret on CREATE.
    """

    enable_auth = False
    name = "llmisvc-tinyllama-s3-connection-cpu"
    storage_uri = "placeholder"
    use_connection = True
    connection_secret_fixture = "s3_connection_secret"
    connection_path = LLMISVC_S3_CONNECTION_PATH

    @classmethod
    def verify_injection(cls, llmisvc: LLMInferenceService) -> None:
        assert cls.connection_secret_name is not None, "connection_secret_name must be bound via with_overrides"
        assert cls.connection_bucket is not None, "connection_bucket must be bound via with_overrides"
        assert cls.connection_path is not None
        assert llmisvc.namespace is not None
        assert_llmisvc_s3_fully_injected(
            client=llmisvc.client,
            llmisvc=llmisvc,
            namespace=llmisvc.namespace,
            secret_name=cls.connection_secret_name,
            bucket=cls.connection_bucket,
            path=cls.connection_path,
        )


class TinyLlamaS3ConnectionSmokeConfig(TinyLlamaS3ConnectionConfig):
    """S3 connection variant that skips the Ready/pod-readiness wait — injection-only smoke check."""

    name = "llmisvc-tinyllama-s3-connection-smoke"
    wait = False


class TinyLlamaHfConnectionConfig(CpuConfig):
    """TinyLlama via a `uri`-typed ConnectionsAPI connection Secret (hf://), CPU inference."""

    enable_auth = False
    name = "llmisvc-tinyllama-hf-connection-cpu"
    storage_uri = "placeholder"
    wait_timeout = 420
    use_connection = True
    connection_secret_fixture = "uri_connection_secret"

    @classmethod
    def verify_injection(cls, llmisvc: LLMInferenceService) -> None:
        assert_llmisvc_uri_injected(llmisvc=llmisvc, expected_uri=ModelStorage.HuggingFace.TINYLLAMA)


class TinyLlamaOciConnectionConfig(CpuConfig):
    """TinyLlama via an `oci`-typed ConnectionsAPI connection Secret, CPU inference.

    Unlike S3/URI, OCI injection never rewrites `spec.model.uri` — it only adds an
    `imagePullSecrets` entry — so `storage_uri` is the real OCI modelcar reference.
    """

    enable_auth = False
    name = "llmisvc-tinyllama-oci-connection-cpu"
    storage_uri = ModelStorage.OCI.TINYLLAMA
    use_connection = True
    connection_secret_fixture = "oci_connection_secret"

    @classmethod
    def verify_injection(cls, llmisvc: LLMInferenceService) -> None:
        assert cls.connection_secret_name is not None, "connection_secret_name must be bound via with_overrides"
        assert_llmisvc_oci_injected(llmisvc=llmisvc, secret_name=cls.connection_secret_name)


class TinyLlamaOciGpuConfig(GpuConfig):
    """TinyLlama via OCI container registry, GPU inference."""

    enable_auth = False
    name = "llmisvc-tinyllama-oci-gpu"
    storage_uri = ModelStorage.OCI.TINYLLAMA
    model_name = ModelName.TINYLLAMA

    @classmethod
    def container_env(cls):
        return super().container_env() + [
            {
                "name": "VLLM_ADDITIONAL_ARGS",
                "value": "--enable-auto-tool-choice --tool-call-parser hermes",
            },
        ]


class TinyLlamaS3GpuConfig(GpuConfig):
    """TinyLlama via S3 bucket, GPU inference."""

    enable_auth = False
    name = "llmisvc-tinyllama-s3-gpu"
    storage_uri = ModelStorage.S3.TINYLLAMA
    model_name = ModelName.TINYLLAMA

    @classmethod
    def container_env(cls):
        return super().container_env() + [
            {
                "name": "VLLM_ADDITIONAL_ARGS",
                "value": "--enable-auto-tool-choice --tool-call-parser hermes",
            },
        ]


class TinyLlamaOciGpuAuthConfig(TinyLlamaOciGpuConfig):
    """TinyLlama via OCI, GPU inference with auth enabled."""

    enable_auth = True
    name = "llmisvc-tinyllama-oci-gpu-auth"


class TinyLlamaHfGpuConfig(GpuConfig):
    """TinyLlama via HuggingFace, GPU inference."""

    enable_auth = False
    name = "llmisvc-tinyllama-hf-gpu"
    storage_uri = ModelStorage.HuggingFace.TINYLLAMA
    model_name = ModelName.TINYLLAMA

    @classmethod
    def container_env(cls):
        return super().container_env() + [
            {
                "name": "VLLM_ADDITIONAL_ARGS",
                "value": "--enable-auto-tool-choice --tool-call-parser hermes",
            },
        ]


class Qwen3MoeDummyGpuConfig(GpuConfig):
    """Qwen3-MoE dummy model via HuggingFace, GPU inference.

    ~20M randomly initialized Qwen3-MoE with 8 experts. Produces garbage output
    but uses a real MoE architecture (qwen3_moe), so vLLM loads and serves it
    correctly. Useful for fast validation of MoE deployment plumbing without
    waiting for a real model to download.
    """

    enable_auth = False
    name = "llmisvc-qwen3-moe-dummy"
    storage_uri = "hf://threcc/qwen3-moe-dummy:93b3d84e2aa41d09bcd473fb8241f6bfa0a0363b"
    model_name = "qwen3-moe-dummy"
