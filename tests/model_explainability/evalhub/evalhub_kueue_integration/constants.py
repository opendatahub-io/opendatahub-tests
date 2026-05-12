EVALHUB_API_BASE = "/api/v1"
EVALHUB_JOBS_ENDPOINT = f"{EVALHUB_API_BASE}/evaluations/jobs"
EVALHUB_HEALTH_ENDPOINT = f"{EVALHUB_API_BASE}/health"

KUEUE_OPERATOR_VERSION = "1.3.0"
KUEUE_OPERATOR_NAMESPACE = "openshift-kueue-operator"

CLUSTER_QUEUE_NAME = "evalhub-test-cq"
LOCAL_QUEUE_NAME = "evalhub-test-queue"
RESOURCE_FLAVOR_NAME = "evalhub-test-flavor"

DEFAULT_CPU_QUOTA = "2"
DEFAULT_MEMORY_QUOTA = "8Gi"
E2E_CPU_QUOTA = "1"
E2E_MEMORY_QUOTA = "4Gi"

SMALL_CPU_QUOTA = "500m"
SMALL_MEMORY_QUOTA = "1Gi"

KUEUE_MANAGED_LABEL = "kueue.openshift.io/managed"
KUEUE_QUEUE_NAME_LABEL = "kueue.x-k8s.io/queue-name"

EVALHUB_TENANT_LABEL = "evalhub.trustyai.opendatahub.io/tenant"


class Timeout:
    JOB_ADMISSION = 120
    JOB_RUNNING = 300
    JOB_COMPLETION = 600
    JOB_PREEMPTION = 120
    KUEUE_READY = 240
    POLL_INTERVAL = 5


class EvalJobState:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkloadConditionType:
    QUOTA_RESERVED = "QuotaReserved"
    ADMITTED = "Admitted"
    EVICTED = "Evicted"
    PREEMPTED = "Preempted"
    REQUEUED = "Requeued"
    FINISHED = "Finished"
