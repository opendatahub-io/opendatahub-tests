from tests.ai_safety.image_constants import AiSafetyImages

# Quota big enough to hold one eval job (~2 CPU / 4Gi) plus the preemptor Job at once,
# so contention lands on the ``pods`` dimension (SINGLE_POD_QUOTA), not CPU/memory.
KUEUE_CPU_QUOTA = "3"
KUEUE_MEMORY_QUOTA = "5Gi"

# EvalHub jobs run at priority 0; this higher-priority class lets a competing job
# preempt them.
HIGH_PRIORITY_VALUE = 1000

# Cap pods at 1 so a second, higher-priority pod forces a preemption.
SINGLE_POD_QUOTA = "1"

# Preemptor Job requests: small enough to fit the quota above, so the pod cap (not
# CPU/memory) is what triggers preemption.
PREEMPTOR_CPU_REQUEST = "1"
PREEMPTOR_MEMORY_REQUEST = "1Gi"

VLLM_EMULATOR = "vllm-emulator"
VLLM_EMULATOR_IMAGE: str = AiSafetyImages.VLLM_EMULATOR
