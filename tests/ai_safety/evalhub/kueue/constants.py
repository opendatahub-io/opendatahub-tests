from tests.ai_safety.image_constants import AiSafetyImages

# Sized to fit exactly one eval job pod: adapter (2 CPU / 4Gi) + sidecar (200m / 512Mi)
# = 2200m / 4.5Gi. A quota of 3 CPU / 5Gi admits one job and keeps a second pending.
KUEUE_CPU_QUOTA = "3"
KUEUE_MEMORY_QUOTA = "5Gi"

# Preemption / priority configuration.
# EvalHub-submitted jobs always carry priority 0; a WorkloadPriorityClass with a
# higher value is used to create a competing job that can preempt them.
HIGH_PRIORITY_VALUE = 1000
# The preemptor requests the whole single-job quota, so it cannot be admitted
# alongside an already-running EvalHub job — forcing a preemption decision.
PREEMPTOR_CPU_REQUEST = KUEUE_CPU_QUOTA
PREEMPTOR_MEMORY_REQUEST = KUEUE_MEMORY_QUOTA

VLLM_EMULATOR = "vllm-emulator"
VLLM_EMULATOR_IMAGE: str = AiSafetyImages.VLLM_EMULATOR
