from tests.ai_safety.image_constants import AiSafetyImages

# Allow enough CPU and memory for both jobs, but limit pods to one, so preemption is
# triggered by pod count.
KUEUE_CPU_QUOTA = "3"
KUEUE_MEMORY_QUOTA = "5Gi"
SINGLE_POD_QUOTA = "1"

# The operator names some tenant RoleBindings '{instance.Name}-{tenant_ns}-...' without the
# instance namespace, so same-named instances that serve the same tenant namespace overwrite
# each other's bindings. The Kueue EvalHub is session-scoped, so it must not share a name with
# the class-scoped 'evalhub-mt' instances used by later tests.
EVALHUB_KUEUE_CR_NAME: str = "evalhub-kueue"

# EvalHub jobs run at priority 0; this higher-priority class lets a competing job
# preempt them.
HIGH_PRIORITY_VALUE = 1000

# Preemptor Job requests, small enough to fit the quota above.
PREEMPTOR_CPU_REQUEST = "1"
PREEMPTOR_MEMORY_REQUEST = "1Gi"

VLLM_EMULATOR = "vllm-emulator"
VLLM_EMULATOR_IMAGE: str = AiSafetyImages.VLLM_EMULATOR
