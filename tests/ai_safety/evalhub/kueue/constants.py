from tests.ai_safety.image_constants import AiSafetyImages

# ClusterQueue quota for the preemption tests. An EvalHub eval pod's adapter container
# requests ~2 CPU / 4Gi (Kueue records this on the Workload), so this quota comfortably
# admits one eval job plus the small competing preemptor Job below (2+1 CPU, 4+1Gi) at
# the same time. Because both fit within cpu/memory, contention is forced on the
# ``pods`` dimension instead (see SINGLE_POD_QUOTA). The shared single/multi-job queues
# don't rely on quota exhaustion at all; they gate via ``stopPolicy: HoldAndDrain``.
KUEUE_CPU_QUOTA = "3"
KUEUE_MEMORY_QUOTA = "5Gi"

# Preemption / priority configuration.
# EvalHub-submitted jobs always carry priority 0; a WorkloadPriorityClass with a
# higher value is used to create a competing job that can preempt them.
HIGH_PRIORITY_VALUE = 1000
# Contention is forced on the Kueue built-in ``pods`` resource: with cpu/memory quota
# sized to hold both the eval job and the preemptor at once, capping ``pods`` at one is
# what makes a second (higher-priority) pod trigger a preemption decision.
SINGLE_POD_QUOTA = "1"
# The competing preemptor Job requests modest CPU/memory that comfortably fit the
# quota above; the single-pod limit — not CPU/memory — is what forces preemption.
PREEMPTOR_CPU_REQUEST = "1"
PREEMPTOR_MEMORY_REQUEST = "1Gi"

VLLM_EMULATOR = "vllm-emulator"
VLLM_EMULATOR_IMAGE: str = AiSafetyImages.VLLM_EMULATOR
