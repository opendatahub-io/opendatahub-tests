import pytest
import requests
from ocp_utilities.monitoring import Prometheus
from simple_logger.logger import get_logger


LOGGER = get_logger(name=__name__)


@pytest.fixture()
def deleted_metrics(prometheus: Prometheus) -> None:
    for metric in ("tgi_request_success", "tgi_request_count"):
        LOGGER.info(f"deleting {metric} metric")
        requests.get(
            f"{prometheus.api_url}/api/v1/admin/tsdb/delete_series?match[]={metric}",
            headers=prometheus.headers,
            verify=prometheus.verify_ssl,
        )
