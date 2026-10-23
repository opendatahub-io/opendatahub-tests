"""Constants for MaaS observability integration tests."""

MAAS_CONFIG_NAME: str = "default"

LIMITADOR_SERVICE_MONITOR_NAME: str = "limitador-metrics"
LIMITADOR_APP_LABEL: str = "limitador"
LIMITADOR_METRICS_PATH: str = "/metrics"
LIMITADOR_METRICS_PORT: str = "http"
DEFAULT_LIMITADOR_SCRAPE_INTERVAL: str = "30s"
LIMITADOR_SCRAPE_LABEL: str = "monitoring.opendatahub.io/scrape"
LIMITADOR_SCRAPE_LABEL_VALUE: str = "true"

USAGE_LOGS_ENVOY_FILTER_NAME: str = "maas-model-access-logs"
USAGE_LOGS_COLLECTOR_NAME: str = "usage-logs"
USAGE_LOGS_CRB_NAME: str = "usage-collector-application-logs-write"

LIMITADOR_UP_METRIC_QUERY: str = "limitador_up"
RHOAI_THANOS_QUERIER_ROUTE_NAME: str = "data-science-thanos-querier-route"

SERVICE_MONITOR_CRD_NAME: str = "servicemonitors.monitoring.coreos.com"
OTEL_COLLECTOR_CRD_NAME: str = "opentelemetrycollectors.opentelemetry.io"

LIMITADOR_DEPLOYMENT_NAMESPACES: tuple[str, ...] = (
    "kuadrant-system",
    "rh-connectivity-link",
)

METRICS_POLL_TIMEOUT: int = 240
LIMITADOR_SERVICE_MONITOR_WAIT_TIMEOUT: int = 120
USAGE_LOGGING_RESOURCES_WAIT_TIMEOUT: int = 120
LIMITADOR_SCRAPE_INTERVAL_TEST_VALUE: str = "1m"

# --- Usage logs stored in the odh-observability LokiStack ---
MONITORING_CRD_NAME: str = "monitorings.services.platform.opendatahub.io"
LOKI_STACK_CRD_NAME: str = "lokistacks.loki.grafana.com"
LOKI_GATEWAY_SERVICE_NAME: str = "data-science-usage-logs-lokistack-gateway-http"
LOKI_GATEWAY_SERVICE_PORT: int = 8080
LOKI_QUERY_LOCAL_PORT: int = 8108
LOKI_QUERY_TENANT: str = "application"
LOKI_QUERY_PATH: str = f"/api/logs/v1/{LOKI_QUERY_TENANT}/loki/api/v1/query"
LOKI_QUERY_URL: str = f"https://localhost:{LOKI_QUERY_LOCAL_PORT}{LOKI_QUERY_PATH}"
LOKI_QUERY_RANGE_WINDOW: str = "1h"
# odh-observability projects the Monitoring CR's usage-logs conditions onto DSCI.status.conditions.
DSCI_USAGE_LOGS_CONDITIONS: tuple[str, ...] = ("Ready", "UsageLogsCollectorAvailable")
DSCI_USAGE_LOGS_CONDITIONS_WAIT_TIMEOUT: int = 600
USAGE_LOGS_PROPAGATION_TIMEOUT: int = 300
USAGE_LOGS_PROPAGATION_SLEEP: int = 15
USAGE_LOGS_INFERENCE_CALLS: int = 3
# Token quota of the dedicated rate-limit subscription: small enough to be exhausted within a
# few inference calls, so the gateway starts rejecting requests with HTTP 429.
USAGE_LOGS_RATE_LIMIT_TOKENS_PER_MINUTE: int = 16
USAGE_LOGS_RATE_LIMIT_MAX_CALLS: int = 10

LOKI_SERVICE_NAME_LABEL: str = "service_name"
LOKI_MODEL_LABEL: str = "model"
LOKI_SUBSCRIPTION_LABEL: str = "subscription"
LOKI_RESPONSE_TYPE_LABEL: str = "response_type"
LOKI_RESPONSE_TYPE_HIT: str = "hit"
LOKI_RESPONSE_TYPE_RATE_LIMIT: str = "rate_limit"
USAGE_LOGS_SERVICE_NAME: str = "models-as-a-service"

# SeaweedFS S3 backend backing the LokiStack object storage
USAGE_LOGS_SEAWEEDFS_NAME: str = "usage-logs-seaweedfs"
USAGE_LOGS_SEAWEEDFS_FILER_CONFIG_NAME: str = f"{USAGE_LOGS_SEAWEEDFS_NAME}-filer-config"
USAGE_LOGS_SEAWEEDFS_CREDENTIALS_SECRET_NAME: str = f"{USAGE_LOGS_SEAWEEDFS_NAME}-credentials"
LOKI_S3_SECRET_NAME: str = "usage-logs-loki-s3"
LOKI_S3_BUCKET_NAME: str = "usage-logs"
LOKI_S3_REGION: str = "us-east-1"


def loki_usage_requests_query(model: str) -> str:
    """Build a LogQL query counting usage-logged inference requests for a model."""
    return (
        f'sum(count_over_time({{{LOKI_SERVICE_NAME_LABEL}="{USAGE_LOGS_SERVICE_NAME}", '
        f'{LOKI_MODEL_LABEL}="{model}"}}[{LOKI_QUERY_RANGE_WINDOW}]))'
    )


def loki_usage_successful_requests_query(model: str) -> str:
    """Build a LogQL query counting usage-logged successful (hit) inference requests for a model."""
    return (
        f'sum(count_over_time({{{LOKI_SERVICE_NAME_LABEL}="{USAGE_LOGS_SERVICE_NAME}", '
        f'{LOKI_MODEL_LABEL}="{model}", {LOKI_RESPONSE_TYPE_LABEL}="{LOKI_RESPONSE_TYPE_HIT}"}}'
        f"[{LOKI_QUERY_RANGE_WINDOW}]))"
    )


def loki_usage_tokens_query(model: str) -> str:
    """Build a LogQL query summing usage-logged total tokens for a model's successful responses."""
    return (
        f'sum(sum_over_time({{{LOKI_SERVICE_NAME_LABEL}="{USAGE_LOGS_SERVICE_NAME}", '
        f'{LOKI_MODEL_LABEL}="{model}", {LOKI_RESPONSE_TYPE_LABEL}="{LOKI_RESPONSE_TYPE_HIT}"}} '
        f"| unwrap tokens_total [{LOKI_QUERY_RANGE_WINDOW}]))"
    )


# Deliberately not filtered by model: a rate-limited call is rejected by the gateway before a
# model response exists, so its `model` label is not guaranteed to match the served model name
# the hit/tokens queries key on. Scope comes from response_type plus the subscription label,
# which the gateway access log takes from the API key's subscription (X-MaaS-Subscription) and
# populates for rejected requests too. The subscription name is unique per test, so unrelated
# 429s in the LokiStack are excluded.
def loki_usage_rate_limited_requests_query(subscription: str) -> str:
    """Build a LogQL query counting usage-logged rate-limited (429) inference requests for a subscription."""
    return (
        f'sum(count_over_time({{{LOKI_SERVICE_NAME_LABEL}="{USAGE_LOGS_SERVICE_NAME}", '
        f'{LOKI_SUBSCRIPTION_LABEL}="{subscription}", '
        f'{LOKI_RESPONSE_TYPE_LABEL}="{LOKI_RESPONSE_TYPE_RATE_LIMIT}"}}'
        f"[{LOKI_QUERY_RANGE_WINDOW}]))"
    )


def limitador_scrape_target_up_query() -> str:
    """Build a PromQL query that confirms Limitador metrics are scraped."""
    namespaces = "|".join(LIMITADOR_DEPLOYMENT_NAMESPACES)
    return f'{LIMITADOR_UP_METRIC_QUERY}{{k8s_namespace_name=~"{namespaces}"}}'
