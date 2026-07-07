"""Fast image configurations for LLMInferenceService resources.

Fast-image CRs are optional LLMInferenceServiceConfig resources deployed by the
RHOAI operator.  They provide pre-optimized vLLM container images for specific
GPU architectures (e.g. NVIDIA CUDA).  CR names follow the pattern
``<version-prefix>kserve-config-llm-…-fast-1`` / ``…-fast-2``.

Subclasses set ``accelerator_config_name_regex`` to a regex that selects the
desired fast CR variant.  All CR discovery is handled by
``GpuConfig._resolve_base_refs`` at build time — subclasses are pure data
classes with no discovery logic.

When the required fast CR is not present on the cluster,
``GpuConfig._resolve_accelerator`` automatically skips the test (via
``pytest.skip``) rather than silently falling back to the default CUDA template.
"""

from .config_models import TinyLlamaOciGpuConfig
from .config_singlenode_prefill_decode import SingleNodePrefillDecodeConfig

# Regexes for matching fast-image LLMInferenceServiceConfig CR names.
# Used by all fast-image config classes to avoid duplicating the pattern.
FAST_1_REGEX = ".*fast-1$"
FAST_2_REGEX = ".*fast-2$"


# config classes for fast images GPU inference.
#
# They override ``accelerator_config_name_regex`` with a regex matching
# the desired fast CR name (e.g. ``.*fast-1$``).  The base ``GpuConfig``
# default regex (``_DEFAULT_ACCELERATOR_CONFIG_NAME_REGEX``) explicitly
# excludes fast CRs, so only subclasses that override it will match them.
#
# When no matching CR is found on the cluster, ``_resolve_accelerator``
# detects that the regex differs from the default and calls ``pytest.skip``
# instead of falling through to the default CUDA template.


class TinyLlamaFast1Config(TinyLlamaOciGpuConfig):
    """TinyLlama via OCI, GPU inference with the fast-1 optimized vLLM image."""

    name = "llmisvc-tinyllama-oci-fast-1"
    accelerator_config_name_regex = FAST_1_REGEX


class TinyLlamaFast2Config(TinyLlamaOciGpuConfig):
    """TinyLlama via OCI, GPU inference with the fast-2 optimized vLLM image."""

    name = "llmisvc-tinyllama-oci-fast-2"
    accelerator_config_name_regex = FAST_2_REGEX


class SingleNodePDFast1Config(SingleNodePrefillDecodeConfig):
    """Single-node P/D with fast-1 optimized vLLM image.

    Inherits all P/D behavior (NixlConnector, pod affinity, 2 GPUs) and
    overrides ``accelerator_config_name_regex`` to select the fast-1 CR.
    Skipped automatically when the fast-1 CR is not present on the cluster.
    """

    name = "llmisvc-singlenode-pd-fast-1"
    accelerator_config_name_regex = FAST_1_REGEX


class SingleNodePDFast2Config(SingleNodePrefillDecodeConfig):
    """Single-node P/D with fast-2 optimized vLLM image.

    Inherits all P/D behavior (NixlConnector, pod affinity, 2 GPUs) and
    overrides ``accelerator_config_name_regex`` to select the fast-2 CR.
    Skipped automatically when the fast-2 CR is not present on the cluster.
    """

    name = "llmisvc-singlenode-pd-fast-2"
    accelerator_config_name_regex = FAST_2_REGEX
