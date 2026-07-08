"""Constants for LLMD tests."""

from utilities.constants import Labels

# List of accelerators supported by llm-d tests
LLMD_TESTS_SUPPORTED_ACCELERATORS = (Labels.Nvidia.NVIDIA_COM_GPU, Labels.ROCm.ROCM_GPU)

# DSC status condition that gates LLMD test execution
LLMD_DSC_CONDITION: str = "KserveLLMInferenceServiceDependencies"

# KServe + LLMISVC controller deployments
LLMD_KSERVE_CONTROLLER_DEPLOYMENTS: list[str] = [
    "kserve-controller-manager",
    "odh-model-controller",
    "llmisvc-controller-manager",
]
