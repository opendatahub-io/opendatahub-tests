from simple_logger.logger import get_logger

import pytest
from ocp_utilities.monitoring import Prometheus

LOGGER = get_logger(name=__name__)


@pytest.mark.order("last")
def test_mr_operator_not_oomkilled(prometheus_for_monitoring: Prometheus):
    result = prometheus_for_monitoring.query_sampler(
        query='kube_pod_container_status_last_terminated_reason{reason="OOMKilled"}'
    )
    if result:
        for entry in result:
            LOGGER.info(entry)
            pod_name = entry["metric"]["pod"]
            if pod_name.startswith("model-registry-operator-controller-manager"):
                pytest.fail(f"Pod {pod_name} was oomkilled: {entry}")
