import shlex

import pytest
from ocp_utilities.monitoring import Prometheus
from pyhelper_utils.shell import run_command


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
