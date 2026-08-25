import re

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
EVALHUB_LOG_COMPLETED_MARKER: str = "Evaluation completed successfully"
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
EVALHUB_CRD_NAME: str = f"{EVALHUB_PLURAL}.{EVALHUB_API_GROUP}"

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
EVALHUB_EVENTS_CLUSTERROLE: str = "trustyai-service-operator-evalhub-events"

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

# Provider IDs for system providers
LM_EVALUATION_HARNESS_PROVIDER_ID: str = "lm_evaluation_harness"

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

# PVC storage test data
PVC_TEST_DATA_NAME: str = "evalhub-test-data"
PVC_TEST_DATA_SIZE: str = "2Gi"
PVC_TOKENIZER_PATH: str = "/test_data/tokenizer"

# Git storage source test data (RHAISTRAT-2058)
# Field names below (test_data_ref.git.*, resolved_sha) follow the test plan's documented
# example payloads; the strategy marks the exact API schema as TBD pending implementation docs.
# Defaults point at eval-hub's own vendored offline lm-eval cache (tests/git-testdata), which
# exists specifically so FVT can exercise git clone/checkout without live Hugging Face downloads;
# see https://github.com/eval-hub/eval-hub/tree/main/tests/git-testdata. Env vars still override.
GIT_PUBLIC_REPO_URL_ENV: str = "GIT_TEST_PUBLIC_REPO_URL"
GIT_PUBLIC_REPO_REF_ENV: str = "GIT_TEST_PUBLIC_REPO_REF"
GIT_PUBLIC_REPO_SUB_PATH_ENV: str = "GIT_TEST_PUBLIC_REPO_SUB_PATH"
GIT_PUBLIC_REPO_URL: str = "https://github.com/eval-hub/eval-hub"
GIT_DEFAULT_REF: str = "main"
GIT_PUBLIC_REPO_SUB_PATH: str = "tests/git-testdata"
# Tokenizer mount path once sub_path narrows the clone to tests/git-testdata (arc_easy cache).
GIT_TOKENIZER_PATH: str = "/test_data/tokenizer"
# Tokenizer path when the full repository is cloned without sub_path narrowing.
GIT_FULL_REPO_TOKENIZER_PATH: str = "/test_data/tests/git-testdata/tokenizer"
# Guaranteed-nonexistent per RFC 2606; used for negative tests that must not resolve.
GIT_NONEXISTENT_REPO_URL: str = "https://git.example.com/does-not-exist/repo.git"
GIT_COMMIT_SHA_PATTERN: re.Pattern[str] = re.compile(r"^[0-9a-f]{7,40}$")

# Hardware profile
EVALHUB_DEFAULT_HARDWARE_PROFILE: str = "default-profile"

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

# Operator Reconciliation Observability (RHAISTRAT-1606 / RHAI-241)

# Prometheus metric names exposed on operator :8080
RECONCILE_DURATION_METRIC: str = "evalhub_controller_reconcile_duration_seconds"
RECONCILE_TOTAL_METRIC: str = "evalhub_controller_reconcile_total"
RECONCILE_ERRORS_METRIC: str = "evalhub_controller_reconcile_errors_total"
MANAGED_INSTANCES_METRIC: str = "evalhub_managed_instances_total"
JOB_FAILURE_EVENTS_METRIC: str = "evalhub_job_failure_events_total"

EVALHUB_RECONCILE_METRICS: tuple[str, ...] = (
    RECONCILE_DURATION_METRIC,
    RECONCILE_TOTAL_METRIC,
    RECONCILE_ERRORS_METRIC,
    MANAGED_INSTANCES_METRIC,
    JOB_FAILURE_EVENTS_METRIC,
)

# Metric label keys
METRIC_LABEL_CONTROLLER: str = "controller"
METRIC_LABEL_RESULT: str = "result"
METRIC_LABEL_ERROR_TYPE: str = "error_type"
METRIC_LABEL_FAILURE_REASON: str = "failure_reason"

# Metric label values — reconciliation results
RESULT_SUCCESS: str = "success"
RESULT_REQUEUE: str = "requeue"
RESULT_ERROR: str = "error"

# Metric label values — bounded error_type enumeration
ERROR_TYPE_DEPLOYMENT_CREATE_FAILED: str = "deployment_create_failed"
ERROR_TYPE_SERVICE_UPDATE_FAILED: str = "service_update_failed"
ERROR_TYPE_OTHER: str = "other"

EVALHUB_ERROR_TYPES: tuple[str, ...] = (
    ERROR_TYPE_DEPLOYMENT_CREATE_FAILED,
    ERROR_TYPE_SERVICE_UPDATE_FAILED,
    ERROR_TYPE_OTHER,
)

# Controller label value used in all metrics
EVALHUB_CONTROLLER_LABEL_VALUE: str = "evalhub"

# Operator metrics port (kube-rbac-proxy)
OPERATOR_METRICS_PORT: int = 8080

# OTEL trace span names emitted by the EvalHub controller
SPAN_RECONCILE: str = "evalhub.reconcile"
SPAN_RECONCILE_CONFIGMAP: str = "evalhub.reconcile.configmap"
SPAN_RECONCILE_DEPLOYMENT: str = "evalhub.reconcile.deployment"
SPAN_RECONCILE_SERVICE: str = "evalhub.reconcile.service"
SPAN_RECONCILE_ROUTE: str = "evalhub.reconcile.route"
SPAN_RECONCILE_RBAC: str = "evalhub.reconcile.rbac"
SPAN_JOB_FAILURE_RECONCILE: str = "evalhub.job_failure_reconcile"

EVALHUB_RECONCILE_CHILD_SPANS: tuple[str, ...] = (
    SPAN_RECONCILE_CONFIGMAP,
    SPAN_RECONCILE_DEPLOYMENT,
    SPAN_RECONCILE_SERVICE,
    SPAN_RECONCILE_ROUTE,
    SPAN_RECONCILE_RBAC,
)

# Span attribute keys
SPAN_ATTR_K8S_NAMESPACE: str = "k8s.namespace"
SPAN_ATTR_EVALHUB_NAME: str = "evalhub.name"
SPAN_ATTR_RECONCILE_GENERATION: str = "reconcile.generation"
SPAN_ATTR_JOB_NAME: str = "job_name"
SPAN_ATTR_FAILURE_REASON: str = "failure_reason"
SPAN_ATTR_EXIT_CODE: str = "exit_code"

# OTEL trace collector for operator reconcile spans
OTEL_TRACE_COLLECTOR_NAMESPACE: str = "otel-trace-collector"
OTEL_TRACE_COLLECTOR_NAME: str = "otel-trace-collector"
OTEL_TRACE_COLLECTOR_LABELS: dict[str, str] = {"app": "otel-trace-collector"}

# Operator pod label selector
OPERATOR_POD_LABEL_SELECTOR: str = "control-plane=controller-manager,app.kubernetes.io/name=trustyai-service-operator"

# Operator service name in OTEL traces
OPERATOR_OTEL_SERVICE_NAME: str = "trustyai-service-operator"
