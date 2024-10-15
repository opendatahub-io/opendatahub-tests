from typing import List

from ocp_resources.maria_db import MariaDB
from ocp_resources.mariadb_operator import MariadbOperator
from ocp_resources.pod import Pod
from ocp_utilities.operators import TIMEOUT_5MIN
from timeout_sampler import TimeoutSampler, TimeoutExpiredError
from simple_logger.logger import get_logger


LOGGER = get_logger(name=__name__)


def wait_for_mariadb_operator_pods(mariadb_operator: MariadbOperator, timeout: int = 300) -> None:
    def _check_if_mariadb_operator_pods_ready() -> bool:
        expected_pods: List[str] = [
            "mariadb-operator",
            "mariadb-operator-cert-controller",
            "mariadb-operator-helm-controller-manager",
            "mariadb-operator-webhook",
        ]

        pods = Pod.get(namespace=mariadb_operator.namespace)

        for pod_prefix in expected_pods:
            matching_pods = [pod for pod in pods if pod.name.startswith(pod_prefix)]

            if matching_pods:
                for pod in matching_pods:
                    try:
                        pod.wait_for_status(status=Pod.Status.RUNNING, timeout=timeout)
                    except TimeoutError:
                        return False
            else:
                LOGGER(f"Waiting for {pod_prefix} pod to be created...")
                return False

        return True

    try:
        for sample in TimeoutSampler(
            wait_timeout=TIMEOUT_5MIN,
            sleep=5,
            func=_check_if_mariadb_operator_pods_ready(),
        ):
            if sample:
                break
    except TimeoutExpiredError:
        LOGGER.error("MariaDB Operator pods are not ready.")
        raise


def wait_for_mariadb_pods(mariadb: MariaDB, timeout: int = 300) -> None:
    def _check_if_mariadb_pods_ready() -> bool:
        namespace = mariadb.namespace
        label_key = "app.kubernetes.io/instance"
        label_value = "mariadb"

        pods = Pod.get(namespace=namespace)
        matching_pods = [pod for pod in pods if pod.labels.get(label_key) == label_value]

        if matching_pods:
            for pod in matching_pods:
                try:
                    pod.wait_for_status(status=Pod.Status.RUNNING, timeout=timeout)
                except TimeoutError:
                    return False
        else:
            LOGGER.info("Waiting for MariaDB pods to be created...")
            return False

        return True

    try:
        for sample in TimeoutSampler(
            wait_timeout=timeout,
            sleep=5,
            func=_check_if_mariadb_pods_ready,
        ):
            if sample:
                break
    except TimeoutExpiredError:
        LOGGER.error("MariaDB pods are not ready.")
        raise