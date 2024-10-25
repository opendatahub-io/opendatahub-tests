from typing import List

from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.maria_db import MariaDB
from ocp_resources.mariadb_operator import MariadbOperator
from ocp_resources.pod import Pod
from timeout_sampler import TimeoutSampler, TimeoutExpiredError
from simple_logger.logger import get_logger

from tests.trustyai.constants import MARIADB

TIMEOUT_5_MIN = 60 * 5

LOGGER = get_logger(name=__name__)


def wait_for_mariadb_operator_deployments(mariadb_operator: MariadbOperator) -> None:
    expected_deployment_names: List[str] = [
        "mariadb-operator",
        "mariadb-operator-cert-controller",
        "mariadb-operator-helm-controller-manager",
        "mariadb-operator-webhook",
    ]

    for name in expected_deployment_names:
        deployment = Deployment(name=name, namespace=mariadb_operator.namespace)
        deployment.wait_for_replicas()


def wait_for_mariadb_pods(client: DynamicClient, mariadb: MariaDB, timeout: int = TIMEOUT_5_MIN) -> None:
    namespace = mariadb.namespace

    def _check_mariadb_pod() -> bool:
        pods = Pod.get(namespace=namespace)
        matching_pods = [
            pod
            for pod in Pod.get(
                dyn_client=client,
                namespace=namespace,
                label_selector=f"app.kubernetes.io/instance={MARIADB}",
            )
        ]
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
    except TimeoutExpiredError as exc:
        raise exc
