"""Constants for LLMD tests."""

from utilities.constants import Labels

# List of accelerators supported by llm-d tests
LLMD_TESTS_SUPPORTED_ACCELERATORS = (Labels.Nvidia.NVIDIA_COM_GPU, Labels.ROCm.ROCM_GPU)

# LLMInferenceServiceConfig CR name for AMD ROCm vLLM image override
AMD_ROCM_TEMPLATE = "kserve-config-llm-template-amd-rocm"

# LLMInferenceServiceConfig CR names for NVIDIA CUDA fast vLLM image overrides
NVIDIA_CUDA_FAST_1_TEMPLATE = "kserve-config-llm-template-nvidia-cuda-fast-1"
NVIDIA_CUDA_FAST_2_TEMPLATE = "kserve-config-llm-template-nvidia-cuda-fast-2"

# DSC status condition that gates LLMD test execution
LLMD_DSC_CONDITION: str = "KserveLLMInferenceServiceDependencies"

# KServe + LLMISVC controller deployments
LLMD_KSERVE_CONTROLLER_DEPLOYMENTS: list[str] = [
    "kserve-controller-manager",
    "odh-model-controller",
    "llmisvc-controller-manager",
]
