import shlex
from simple_logger.logger import get_logger

import pytest
from ocp_utilities.monitoring import Prometheus
from pyhelper_utils.shell import run_command

LOGGER = get_logger(name=__name__)


def get_prometheus_k8s_token(duration: str = "1800s") -> str:
    token_command = f"oc create token prometheus-k8s -n openshift-monitoring --duration={duration}"
    command_success, out, _ = run_command(command=shlex.split(token_command), verify_stderr=False)
    assert command_success, f"Command {token_command} failed to execute"
    return out


@pytest.fixture(scope="session")
def prometheus_for_monitoring() -> Prometheus:
    return Prometheus(
        verify_ssl=False,
        bearer_token=get_prometheus_k8s_token(duration="86400s"),
    )


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
