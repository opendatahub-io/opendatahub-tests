EVALHUB_SERVICE_NAME: str = "evalhub"
EVALHUB_SERVICE_PORT: int = 8443
EVALHUB_CONTAINER_PORT: int = 8080
EVALHUB_HEALTH_PATH: str = "/api/v1/health"
EVALHUB_METRICS_PATH: str = "/metrics"
EVALHUB_PROVIDERS_PATH: str = "/api/v1/evaluations/providers"
EVALHUB_JOBS_PATH: str = "/api/v1/evaluations/jobs"
EVALHUB_HEALTH_STATUS_HEALTHY: str = "healthy"

EVALHUB_APP_LABEL: str = "eval-hub"
EVALHUB_CONTAINER_NAME: str = "evalhub"
EVALHUB_COMPONENT_LABEL: str = "api"

# CRD details
EVALHUB_API_GROUP: str = "trustyai.opendatahub.io"
EVALHUB_API_VERSION: str = "v1alpha1"
EVALHUB_KIND: str = "EvalHub"
EVALHUB_PLURAL: str = "evalhubs"

# Multi-tenancy
EVALHUB_TENANT_LABEL_KEY: str = "evalhub.trustyai.opendatahub.io/tenant"
EVALHUB_TENANT_LABEL_VALUE: str = "true"
EVALHUB_COLLECTIONS_PATH: str = "/api/v1/evaluations/collections"
EVALHUB_PROVIDERS_ACCESS_CLUSTERROLE: str = "trustyai-service-operator-evalhub-providers-access"
EVALHUB_MT_CR_NAME: str = "evalhub-mt"
EVALHUB_VLLM_EMULATOR_PORT: int = 8000

# ClusterRole names (kustomize namePrefix applied by operator install)
EVALHUB_JOBS_WRITER_CLUSTERROLE: str = "trustyai-service-operator-evalhub-jobs-writer"
EVALHUB_JOB_CONFIG_CLUSTERROLE: str = "trustyai-service-operator-evalhub-job-config"

# EvalHub Kubernetes runtime (batch Job / ConfigMap) — mirrors eval-hub job_builders.go
EVALHUB_K8S_LABEL_APP: str = "app"
EVALHUB_K8S_LABEL_APP_VALUE: str = "evalhub"
EVALHUB_K8S_LABEL_COMPONENT: str = "component"
EVALHUB_K8S_LABEL_COMPONENT_VALUE: str = "evaluation-job"
EVALHUB_K8S_LABEL_JOB_ID: str = "job_id"
EVALHUB_K8S_ANNOTATION_JOB_ID: str = "eval-hub.github.io/job_id"
EVALHUB_K8S_ANNOTATION_PROVIDER_ID: str = "eval-hub.github.io/provider_id"
EVALHUB_K8S_ANNOTATION_BENCHMARK_ID: str = "eval-hub.github.io/benchmark_id"

# Shared RBAC rules for EvalHub user access
EVALHUB_USER_ROLE_RULES: list[dict[str, list[str]]] = [
    {
        "apiGroups": ["trustyai.opendatahub.io"],
        "resources": ["evaluations", "collections", "providers"],
        "verbs": ["get", "list", "create", "update", "delete"],
    },
    {
        "apiGroups": ["mlflow.kubeflow.org"],
        "resources": ["experiments"],
        "verbs": ["create", "get"],
    },
]

# Garak provider
GARAK_PROVIDER_ID: str = "garak-kfp"
GARAK_BENCHMARK_ID: str = "intents"
GARAK_JOB_TIMEOUT: int = 1800  # 30 minutes
GARAK_JOB_POLL_INTERVAL: int = 30  # seconds

# Job service account naming
EVALHUB_JOB_SA_PREFIX: str = "evalhub-"
EVALHUB_JOB_SA_SUFFIX: str = "-job"

# Garak intents CSV
GARAK_INTENTS_S3_KEY: str = "intents/misinformation_prompts.csv"
MINIO_MC_IMAGE = "quay.io/minio/mc@sha256:470f5546b596e16c7816b9c3fa7a78ce4076bb73c2c73f7faeec0c8043923123"
MINIO_UPLOADER_SECURITY_CONTEXT = {
    "allowPrivilegeEscalation": False,
    "capabilities": {"drop": ["ALL"]},
    "runAsNonRoot": True,
    "seccompProfile": {"type": "RuntimeDefault"},
}
