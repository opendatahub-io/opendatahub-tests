"""Fast image configurations for LLMInferenceService resources.

Uses LLMInferenceServiceConfig baseRefs to deploy models with
accelerator-specific fast vLLM image overrides, discovered dynamically
from the cluster.
"""

from kubernetes.dynamic import DynamicClient

from tests.model_serving.model_server.llmd.utils import discover_fast_cr

from .config_models import TinyLlamaOciGpuConfig


class FastImageConfig(TinyLlamaOciGpuConfig):
    """Base class for fast image GPU inference.

    Subclasses set ``fast_suffix`` to the fast variant suffix (e.g. ``-fast-1``).
    ``build()`` detects the cluster accelerator, then discovers the matching
    fast LLMInferenceServiceConfig CR from the cluster.
    Use the ``skip_if_fast_cr_missing`` fixture to skip when no fast CR
    matches the cluster's accelerator.
    """

    fast_suffix: str = ""

    @classmethod
    def build(cls, client: DynamicClient) -> type:
        """Detect GPU accelerator and discover the matching fast CR."""
        resolved = cls._resolve_accelerator(client=client)
        cr_name = discover_fast_cr(
            client=client,
            fast_suffix=cls.fast_suffix,
            accelerator=resolved.accelerator,
        )
        base_refs = [{"name": cr_name}] if cr_name else resolved.base_refs or []
        return resolved.with_overrides(base_refs=base_refs)


class TinyLlamaFast1Config(FastImageConfig):
    """TinyLlama via OCI, GPU inference with fast-1 image."""

    name = "llmisvc-tinyllama-oci-fast-1"
    fast_suffix = "-fast-1"


class TinyLlamaFast2Config(FastImageConfig):
    """TinyLlama via OCI, GPU inference with fast-2 image."""

    name = "llmisvc-tinyllama-oci-fast-2"
    fast_suffix = "-fast-2"
