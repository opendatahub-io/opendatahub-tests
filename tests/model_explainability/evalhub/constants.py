EVALHUB_SERVICE_NAME: str = "evalhub"
EVALHUB_SERVICE_PORT: int = 8443
EVALHUB_CONTAINER_PORT: int = 8080
EVALHUB_HEALTH_PATH: str = "/api/v1/health"
EVALHUB_PROVIDERS_PATH: str = "/api/v1/evaluations/providers"
EVALHUB_HEALTH_STATUS_HEALTHY: str = "healthy"

EVALHUB_APP_LABEL: str = "eval-hub"

# CRD details
EVALHUB_API_GROUP: str = "trustyai.opendatahub.io"
EVALHUB_API_VERSION: str = "v1alpha1"
EVALHUB_KIND: str = "EvalHub"
EVALHUB_PLURAL: str = "evalhubs"

# RBAC ClusterRole names (must match operator config/rbac/evalhub/ YAML files)
EVALHUB_PROVIDERS_ACCESS_CLUSTER_ROLE: str = "trustyai-service-operator-evalhub-providers-access"
