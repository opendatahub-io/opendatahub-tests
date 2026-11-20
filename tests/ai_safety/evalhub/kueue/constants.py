from tests.ai_safety.image_constants import AiSafetyImages

# Sized to fit exactly one eval job pod: adapter (2 CPU / 4Gi) + sidecar (200m / 512Mi)
# = 2200m / 4.5Gi. A quota of 3 CPU / 5Gi admits one job and keeps a second pending.
KUEUE_CPU_QUOTA = "3"
KUEUE_MEMORY_QUOTA = "5Gi"

VLLM_EMULATOR = "vllm-emulator"
VLLM_EMULATOR_IMAGE: str = AiSafetyImages.VLLM_EMULATOR
