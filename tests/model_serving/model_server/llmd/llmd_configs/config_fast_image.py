"""Fast image configurations for LLMInferenceService resources.

Uses LLMInferenceServiceConfig baseRefs to deploy models with the
nvidia-cuda fast-1 or fast-2 vLLM image overrides.
"""

import pytest
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import NotFoundError, ResourceNotFoundError

from tests.model_serving.model_server.llmd.constants import (
    NVIDIA_CUDA_FAST_1_TEMPLATE,
    NVIDIA_CUDA_FAST_2_TEMPLATE,
)
from utilities.constants import Labels

from .config_models import TinyLlamaOciGpuConfig

_LLMISVC_CONFIG_API_VERSION = "serving.kserve.io/v1alpha1"
_LLMISVC_CONFIG_KIND = "LLMInferenceServiceConfig"


class FastImageConfig(TinyLlamaOciGpuConfig):
    """Base class for fast image GPU inference.

    Subclasses set ``fast_template`` to the unversioned LLMInferenceServiceConfig
    CR name.  ``build()`` resolves the version prefix and skips the test when the
    CR does not exist on the cluster.
    """

    fast_template: str = ""

    # fast images are NVIDIA-only
    supported_accelerators = (Labels.Nvidia.NVIDIA_COM_GPU,)

    @classmethod
    def build(cls, client: DynamicClient) -> type:
        """Detect GPU accelerator, resolve the versioned fast template, and skip if absent."""
        resolved = cls._resolve_accelerator(client=client)
        base_refs = cls._resolve_base_refs(client=client, template_name=cls.fast_template)
        cr_name = base_refs[0]["name"]
        cls._skip_if_cr_missing(client=client, cr_name=cr_name)
        return resolved.with_overrides(base_refs=base_refs)

    @staticmethod
    def _skip_if_cr_missing(client: DynamicClient, cr_name: str) -> None:
        """Skip the test if the LLMInferenceServiceConfig CR does not exist on the cluster."""
        try:
            api = client.resources.get(
                api_version=_LLMISVC_CONFIG_API_VERSION,
                kind=_LLMISVC_CONFIG_KIND,
            )
            api.get(name=cr_name)
        except (NotFoundError, ResourceNotFoundError):  # fmt: skip
            pytest.skip(f"LLMInferenceServiceConfig CR '{cr_name}' not found on cluster — skipping test")


class TinyLlamaFast1Config(FastImageConfig):
    """TinyLlama via OCI, GPU inference with nvidia-cuda fast-1 image."""

    name = "llmisvc-tinyllama-oci-fast-1"
    fast_template = NVIDIA_CUDA_FAST_1_TEMPLATE


class TinyLlamaFast2Config(FastImageConfig):
    """TinyLlama via OCI, GPU inference with nvidia-cuda fast-2 image."""

    name = "llmisvc-tinyllama-oci-fast-2"
    fast_template = NVIDIA_CUDA_FAST_2_TEMPLATE
