from tests.ai_safety.image_constants import AiSafetyImages

MINIO_MC_IMAGE: str = AiSafetyImages.MINIO_MC

EVALHUB_SERVICE_NAME: str = "evalhub"
EVALHUB_SERVICE_PORT: int = 8443
EVALHUB_CONTAINER_PORT: int = 8080
EVALHUB_HEALTH_PATH: str = "/api/v1/health"
EVALHUB_METRICS_PATH: str = "/metrics"
EVALHUB_PROVIDERS_PATH: str = "/api/v1/evaluations/providers"
EVALHUB_JOBS_PATH: str = "/api/v1/evaluations/jobs"
EVALHUB_JOB_LOGS_PATH_TEMPLATE: str = "/api/v1/evaluations/jobs/{job_id}/logs"
EVALHUB_JOB_BENCHMARK_LOGS_PATH_TEMPLATE: str = "/api/v1/evaluations/jobs/{job_id}/benchmarks/{benchmark_index}/logs"
EVALHUB_HEALTH_STATUS_HEALTHY: str = "healthy"

# Job log API (RHAISTRAT-1437 / eval-hub HTTP API)
EVALHUB_LOG_CONTENT_TYPE: str = "text/plain"
EVALHUB_LOG_SECTION_PREFIX: str = "=== pod="
EVALHUB_LOG_ADAPTER_CONTAINER: str = "adapter"
EVALHUB_LOG_COMPLETED_MARKER: str = "EVALUATION COMPLETE"
EVALHUB_LOG_DEFAULT_TAIL_LINES: int = 1000
EVALHUB_LOG_MAX_TAIL_LINES: int = 10000

EVALHUB_APP_LABEL: str = "eval-hub"
EVALHUB_CONTAINER_NAME: str = "evalhub"
EVALHUB_KUBE_RBAC_PROXY_CONTAINER: str = "kube-rbac-proxy"
EVALHUB_COMPONENT_LABEL: str = "api"

# CRD details
EVALHUB_API_GROUP: str = "trustyai.opendatahub.io"
EVALHUB_API_VERSION_V1: str = "v1"
EVALHUB_API_VERSION_V1ALPHA1: str = "v1alpha1"
EVALHUB_FULL_API_VERSION_V1: str = f"{EVALHUB_API_GROUP}/v1"
EVALHUB_FULL_API_VERSION_V1ALPHA1: str = f"{EVALHUB_API_GROUP}/v1alpha1"
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
GARAK_SIMPLE_PROVIDER_ID: str = "garak"
GARAK_PROVIDER_ID: str = "garak-kfp"
GARAK_BENCHMARK_ID: str = "intents"
GARAK_QUICK_BENCHMARK_ID: str = "quick"
GARAK_JOB_TIMEOUT: int = 1800  # 30 minutes
GARAK_JOB_POLL_INTERVAL: int = 30  # seconds

# Job service account naming
EVALHUB_JOB_SA_PREFIX: str = "evalhub-"
EVALHUB_JOB_SA_SUFFIX: str = "-job"

# Garak intents CSV
GARAK_INTENTS_S3_KEY: str = "intents/misinformation_prompts.csv"
MINIO_UPLOADER_SECURITY_CONTEXT = {
    "allowPrivilegeEscalation": False,
    "capabilities": {"drop": ["ALL"]},
    "runAsNonRoot": True,
    "seccompProfile": {"type": "RuntimeDefault"},
}

# Minimal MinIO for simple-mode intents (no DSPA needed)
SIMPLE_MINIO_ACCESS_KEY: str = "minioadmin"
SIMPLE_MINIO_SECRET_KEY: str = "minioadmin"
SIMPLE_MINIO_BUCKET: str = "evalhub-data"

# ServiceMonitor and metrics Service
EVALHUB_METRICS_SERVICE_SUFFIX: str = "-metrics"
EVALHUB_METRICS_PORT: int = 8081
EVALHUB_METRICS_COMPONENT_LABEL: str = "metrics"
EVALHUB_SCRAPE_INTERVAL: str = "30s"

# OTEL Collector constants
OTEL_COLLECTOR_NAMESPACE: str = "otel-collector"
OTEL_COLLECTOR_GRPC_PORT: int = 4317
OTEL_COLLECTOR_HTTP_PORT: int = 4318
OTEL_COLLECTOR_PROMETHEUS_PORT: int = 8889

# OTEL error patterns that indicate initialization failure
OTEL_ERROR_PATTERNS: tuple[str, ...] = (
    "failed to initialize meter",
    "meter provider error",
    "panic",
    "OTEL initialization failed",
)

# OTLP export indicators in collector logs
OTLP_INDICATORS: tuple[str, ...] = (
    "ResourceMetrics",
    "ScopeMetrics",
    "http.server.request",
    "github.com/eval-hub",
)
