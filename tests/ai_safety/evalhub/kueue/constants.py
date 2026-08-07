from tests.ai_safety.image_constants import AiSafetyImages

# Kueue Configuration
SINGLE_JOB_CPU_QUOTA = "2"
SINGLE_JOB_MEMORY_QUOTA = "4Gi"
MULTI_JOB_CPU_QUOTA = "8"
MULTI_JOB_MEMORY_QUOTA = "16Gi"

# vLLM emulator configuration
VLLM_EMULATOR = "vllm-emulator"
VLLM_EMULATOR_IMAGE: str = AiSafetyImages.VLLM_EMULATOR
