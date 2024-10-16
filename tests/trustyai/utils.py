from time import sleep
from typing import List

from ocp_resources.deployment import Deployment
from ocp_resources.maria_db import MariaDB
from ocp_resources.mariadb_operator import MariadbOperator
from ocp_resources.pod import Pod
from timeout_sampler import TimeoutSampler, TimeoutExpiredError
from simple_logger.logger import get_logger

LOGGER = get_logger(name=__name__)

def wait_for_mariadb_operator_deployments(mariadb_operator: MariadbOperator, timeout: int = 300) -> None:
    expected_deployment_names: List[str] = [
        "mariadb-operator",
        "mariadb-operator-cert-controller",
        "mariadb-operator-helm-controller-manager",
        "mariadb-operator-webhook",
    ]

    for name in expected_deployment_names:
        deployment = Deployment(name=name, namespace=mariadb_operator.namespace)
        deployment.wait_for_replicas()

def wait_for_mariadb_pods(mariadb: MariaDB, timeout: int = 300) -> None:
    namespace = mariadb.namespace
    label_key = "app.kubernetes.io/instance"
    label_value = "mariadb"

    def _check_mariadb_pod():
        pods = Pod.get(namespace=namespace)
        matching_pods = [pod for pod in pods if pod.labels.get(label_key) == label_value]
        if not matching_pods:
            return False
        pod = matching_pods[0]
        return pod.status == Pod.Status.RUNNING

    sampler = TimeoutSampler(
        wait_timeout=timeout,
        sleep=5,
        func=_check_mariadb_pod,
    )

    try:
        for result in sampler:
            if result:
                return
    except TimeoutExpiredError:
        raise TimeoutError(f"Timed out waiting for MariaDB pod after {timeout} seconds")