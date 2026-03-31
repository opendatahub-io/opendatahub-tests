EVALHUB_SERVICE_NAME: str = "evalhub"
EVALHUB_SERVICE_PORT: int = 8443
EVALHUB_CONTAINER_PORT: int = 8080
EVALHUB_HEALTH_PATH: str = "/api/v1/health"
EVALHUB_PROVIDERS_PATH: str = "/api/v1/evaluations/providers"
EVALHUB_JOBS_PATH: str = "/api/v1/evaluations/jobs"
EVALHUB_BENCHMARKS_PATH: str = "/api/v1/evaluations/benchmarks"
EVALHUB_COLLECTIONS_PATH: str = "/api/v1/evaluations/collections"
EVALHUB_HEALTH_STATUS_HEALTHY: str = "healthy"

EVALHUB_APP_LABEL: str = "eval-hub"

# CRD details
EVALHUB_API_GROUP: str = "trustyai.opendatahub.io"
EVALHUB_API_VERSION: str = "v1alpha1"
EVALHUB_KIND: str = "EvalHub"
EVALHUB_PLURAL: str = "evalhubs"

# RBAC
EVALHUB_PROVIDERS_ACCESS_CLUSTER_ROLE: str = "trustyai-service-operator-evalhub-providers-access"
EVALHUB_EVALUATOR_ROLE: str = "evalhub-evaluator"
EVALHUB_TENANT_LABEL: str = "evalhub.trustyai.opendatahub.io/tenant"

# Multi-tenancy test constants
TENANT_A_NAME: str = "evalhub-tenant-a"
TENANT_B_NAME: str = "evalhub-tenant-b"
TENANT_A_SA_NAME: str = "team-a-user"
TENANT_B_SA_NAME: str = "team-b-user"
TENANT_UNAUTHORISED_SA_NAME: str = "evalhub-no-access-user"
