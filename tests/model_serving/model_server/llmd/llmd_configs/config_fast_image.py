"""Fast image configurations for LLMInferenceService resources.

These configs deploy an LLMInferenceService with baseRefs pointing to
nvidia-cuda fast-1 or fast-2 LLMInferenceServiceConfig CRs. The
LLMInferenceServiceConfig CR overrides the default vLLM container image
with a fast-channel build.
"""

from utilities.constants import Labels

from .config_models import TinyLlamaOciGpuConfig


class Fast1Config(TinyLlamaOciGpuConfig):
    """TinyLlama OCI model on NVIDIA GPU using the fast-1 vLLM image."""

    name = "llmisvc-fast-1"
    base_refs_template = "kserve-config-llm-template-nvidia-cuda-fast-1"

    # GPU requirements
    # default is min_gpus_per_node = 1
    # default is min_nodes = 1
    supported_accelerators = (Labels.Nvidia.NVIDIA_COM_GPU,)


class Fast2Config(TinyLlamaOciGpuConfig):
    """TinyLlama OCI model on NVIDIA GPU using the fast-2 vLLM image."""

    name = "llmisvc-fast-2"
    base_refs_template = "kserve-config-llm-template-nvidia-cuda-fast-2"

    # GPU requirements
    # default is min_gpus_per_node = 1
    # default is min_nodes = 1
    supported_accelerators = (Labels.Nvidia.NVIDIA_COM_GPU,)
