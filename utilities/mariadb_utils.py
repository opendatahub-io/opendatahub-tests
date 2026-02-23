from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.maria_db import MariaDB
from ocp_resources.mariadb_operator import MariadbOperator
from ocp_resources.pod import Pod
from timeout_sampler import TimeoutSampler

from utilities.constants import Timeout


def wait_for_mariadb_pods(client: DynamicClient, mariadb: MariaDB, timeout: int = Timeout.TIMEOUT_5MIN) -> None:
    def _get_mariadb_pods() -> list[Pod]:
        _pods = [
            _pod
            for _pod in Pod.get(
                client=client,
                namespace=mariadb.namespace,
                label_selector=f"app.kubernetes.io/instance={mariadb.name}",
            )
        ]
        return _pods

    sampler = TimeoutSampler(wait_timeout=timeout, sleep=1, func=lambda: bool(_get_mariadb_pods()))

    for sample in sampler:
        if sample:
            break

    pods = _get_mariadb_pods()
    for pod in pods:
        pod.wait_for_condition(
            condition=Pod.Condition.READY,
            status="True",
        )


def wait_for_mariadb_operator_deployments(mariadb_operator: MariadbOperator, client: DynamicClient) -> None:
    expected_deployment_names: list[str] = [
        "mariadb-operator",
        "mariadb-operator-cert-controller",
        "mariadb-operator-helm-controller-manager",
        "mariadb-operator-webhook",
    ]

    for name in expected_deployment_names:
        deployment = Deployment(client=client, name=name, namespace=mariadb_operator.namespace)
        deployment.wait_for_replicas()
