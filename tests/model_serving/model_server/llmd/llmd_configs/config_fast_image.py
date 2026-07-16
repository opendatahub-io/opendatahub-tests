"""Fast image configurations for LLMInferenceService resources.

Fast-image CRs are optional LLMInferenceServiceConfig resources deployed by the
RHOAI operator.  They provide pre-optimized vLLM container images for specific
GPU architectures (e.g. NVIDIA CUDA).  CR names follow the pattern
``<version-prefix>kserve-config-llm-…-fast-1`` / ``…-fast-2``.

Subclasses override ``accelerator_config_name_regex`` with a positive regex
that selects the desired fast CR variant.  The base ``GpuConfig._select_base_refs``
detects the non-default regex and calls ``pytest.skip`` (instead of ``pytest.fail``)
when the CR is not present on the cluster.
"""

from .config_models import TinyLlamaOciGpuConfig
from .config_singlenode_prefill_decode import SingleNodePrefillDecodeConfig

FAST_1_REGEX = ".*fast-1$"
FAST_2_REGEX = ".*fast-2$"


class TinyLlamaFast1Config(TinyLlamaOciGpuConfig):
    """TinyLlama via OCI, GPU inference with the fast-1 optimized vLLM image."""

    name = "llmisvc-tinyllama-oci-fast-1"
    accelerator_config_name_regex = FAST_1_REGEX
    optional_base_refs = True


class TinyLlamaFast2Config(TinyLlamaOciGpuConfig):
    """TinyLlama via OCI, GPU inference with the fast-2 optimized vLLM image."""

    name = "llmisvc-tinyllama-oci-fast-2"
    accelerator_config_name_regex = FAST_2_REGEX
    optional_base_refs = True


class SingleNodePDFast1Config(SingleNodePrefillDecodeConfig):
    """Single-node P/D with fast-1 optimized vLLM image."""

    name = "llmisvc-singlenode-pd-fast-1"
    accelerator_config_name_regex = FAST_1_REGEX
    optional_base_refs = True


class SingleNodePDFast2Config(SingleNodePrefillDecodeConfig):
    """Single-node P/D with fast-2 optimized vLLM image."""

    name = "llmisvc-singlenode-pd-fast-2"
    accelerator_config_name_regex = FAST_2_REGEX
    optional_base_refs = True
