"""Fast image configurations for LLMInferenceService resources.

Fast-image CRs are optional LLMInferenceServiceConfig resources deployed by the
RHOAI operator.  They provide pre-optimized vLLM container images for specific
GPU architectures (e.g. NVIDIA CUDA).  CR names follow the pattern
``<version-prefix>kserve-config-llm-…-fast-1`` / ``…-fast-2``.

Subclasses set ``accelerator_config_name_regex`` to a regex that selects the
desired fast CR variant.  All CR discovery is handled by
``GpuConfig._resolve_base_refs`` at build time — subclasses are pure data
classes with no discovery logic.

Tests using these configs should be decorated with
``@pytest.mark.usefixtures("skip_if_fast_cr_missing")`` so that missing fast
CRs cause a clean skip rather than a silent fallback to the default CUDA
template.
"""

from .config_models import TinyLlamaOciGpuConfig


class FastImageConfig(TinyLlamaOciGpuConfig):
    """Base class for fast image GPU inference.

    Subclasses override ``accelerator_config_name_regex`` with a regex matching
    the desired fast CR name (e.g. ``.*fast-1$``).  The base ``GpuConfig``
    default regex ``^(?!.*fast-)`` explicitly excludes fast CRs, so only
    subclasses that override it will match them.

    Use the ``skip_if_fast_cr_missing`` fixture to skip when no matching CR
    is found on the cluster.  Without it, the test would silently deploy with
    the default CUDA template (because fast CRs use NVIDIA GPUs, which have a
    built-in fallback).
    """


class TinyLlamaFast1Config(FastImageConfig):
    """TinyLlama via OCI, GPU inference with the fast-1 optimized vLLM image."""

    name = "llmisvc-tinyllama-oci-fast-1"
    accelerator_config_name_regex = ".*fast-1$"


class TinyLlamaFast2Config(FastImageConfig):
    """TinyLlama via OCI, GPU inference with the fast-2 optimized vLLM image."""

    name = "llmisvc-tinyllama-oci-fast-2"
    accelerator_config_name_regex = ".*fast-2$"
