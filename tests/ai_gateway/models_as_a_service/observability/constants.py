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


def limitador_scrape_target_up_query() -> str:
    """Build a PromQL query that confirms Limitador metrics are scraped."""
    namespaces = "|".join(LIMITADOR_DEPLOYMENT_NAMESPACES)
    return f'{LIMITADOR_UP_METRIC_QUERY}{{k8s_namespace_name=~"{namespaces}"}}'
