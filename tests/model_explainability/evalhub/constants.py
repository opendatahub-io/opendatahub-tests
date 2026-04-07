EVALHUB_SERVICE_NAME: str = "evalhub"
EVALHUB_SERVICE_PORT: int = 8443
EVALHUB_CONTAINER_PORT: int = 8080
EVALHUB_HEALTH_PATH: str = "/api/v1/health"
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
EVALHUB_COLLECTIONS_PATH: str = "/api/v1/evaluations/collections"
EVALHUB_PROVIDERS_ACCESS_CLUSTERROLE: str = "trustyai-service-operator-evalhub-providers-access"
EVALHUB_MT_CR_NAME: str = "evalhub-mt"
EVALHUB_VLLM_EMULATOR_PORT: int = 8000

# ClusterRole names (kustomize namePrefix applied by operator install)
EVALHUB_JOBS_WRITER_CLUSTERROLE: str = "trustyai-service-operator-evalhub-jobs-writer"
EVALHUB_JOB_CONFIG_CLUSTERROLE: str = "trustyai-service-operator-evalhub-job-config"
