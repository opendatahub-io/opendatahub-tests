"""Constants for serving runtime image validation tests."""

from utilities.constants import RuntimeTemplates

# Placeholder storage URI so the controller creates Deployment/Pod with runtime image.
# No actual model or inference is required; pod phase does not need to be Ready.
PLACEHOLDER_STORAGE_URI = "s3://dummy-bucket/dummy/"

# Runtime configs: display name (for "name : passed") and template name.
# For each we create ServingRuntime + InferenceService, wait for pod(s), validate, then teardown.
RUNTIME_CONFIGS = [
    {"name": "odh_openvino_model_server_image", "template": RuntimeTemplates.OVMS_KSERVE},
    {"name": "odh_vllm_cpu_image", "template": RuntimeTemplates.VLLM_CPU_x86},
    {"name": "odh_vllm_gaudi_image", "template": RuntimeTemplates.VLLM_GAUDI},
    {"name": "odh_mlserver_image", "template": RuntimeTemplates.MLSERVER},
    {"name": "rhaiis_vllm_cuda_image", "template": RuntimeTemplates.VLLM_CUDA},
    {"name": "rhaiis_vllm_rocm_image", "template": RuntimeTemplates.VLLM_ROCM},
    {"name": "rhaiis_vllm_spyre_image", "template": RuntimeTemplates.VLLM_SPYRE},
]
