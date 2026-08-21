# Lifecycle phase label applied to evaluation batch Jobs
LIFECYCLE_PHASE_LABEL = "trustyai.opendatahub.io/evaluation-phase"

# Lifecycle status annotation applied to evaluation batch Jobs (JSON payload)
LIFECYCLE_STATUS_ANNOTATION = "trustyai.opendatahub.io/evaluation-status"

# Kubernetes Event reason codes (CamelCase per K8s convention)
LIFECYCLE_REASON_STARTED = "EvaluationStarted"
LIFECYCLE_REASON_COMPLETED = "EvaluationCompleted"
LIFECYCLE_REASON_FAILED = "EvaluationFailed"
LIFECYCLE_REASON_THRESHOLD_VIOLATED = "EvaluationThresholdViolated"
LIFECYCLE_EXPECTED_REASONS: frozenset[str] = frozenset({
    LIFECYCLE_REASON_STARTED,
    LIFECYCLE_REASON_COMPLETED,
    LIFECYCLE_REASON_FAILED,
    LIFECYCLE_REASON_THRESHOLD_VIOLATED,
})

# Event source.component values
LIFECYCLE_SOURCE_SERVER = "evalhub-server"
LIFECYCLE_SOURCE_OPERATOR = "trustyai-service-operator"

# Phase label values
LIFECYCLE_PHASE_RUNNING = "Running"
LIFECYCLE_PHASE_SUCCEEDED = "Succeeded"
LIFECYCLE_PHASE_FAILED = "Failed"
LIFECYCLE_PHASE_THRESHOLD_VIOLATED = "ThresholdViolated"

# Control plane namespace — EvalHub CR and Deployment live here
LIFECYCLE_SIGNALS_CP_NAMESPACE = "test-k8s-lifecycle-signals-cp"

# EvalHub CR name — distinct from production 'evalhub' and multitenancy 'evalhub-mt'
# to avoid RoleBinding name collisions (operator uses Get-or-Create keyed on CR name)
LIFECYCLE_SIGNALS_CR_NAME = "evalhub-ls"

# Tenant namespace — workloads/jobs run here (labelled with tenant label)
LIFECYCLE_SIGNALS_NAMESPACE = "test-k8s-lifecycle-signals"

# Event emission SLA (seconds) per acceptance criteria
LIFECYCLE_EVENT_EMISSION_TIMEOUT = 30

# Timeouts
LIFECYCLE_JOB_LABEL_TIMEOUT = 120
LIFECYCLE_JOB_SUBMIT_TIMEOUT = 600

# Infrastructure failure constants
LIFECYCLE_OOM_MEMORY_LIMIT = "10Mi"
LIFECYCLE_BAD_IMAGE = "quay.io/trustyai/eval-adapter:does-not-exist"  # noqa: IMG001

# lm_evaluation_harness k8s runtime entrypoint (matches eval-hub bundled provider)
LIFECYCLE_LM_EVAL_K8S_ENTRYPOINT: list[str] = [
    "/opt/app-root/bin/python",
    "/opt/app-root/src/main.py",
]

# Threshold that always fails when using the vLLM emulator (emulator returns random outputs)
LIFECYCLE_THRESHOLD_ACCURACY_HIGH = 1.01

# Threshold the vLLM emulator always meets (acc_norm >= 0.0); overrides provider default 0.25
LIFECYCLE_THRESHOLD_ACCURACY_PASS = 0.0

# Status annotation size limit (Kubernetes annotation size limit is 256 KB)
LIFECYCLE_STATUS_ANNOTATION_MAX_BYTES = 262144
