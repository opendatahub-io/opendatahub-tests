from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError, ResourceNotUniqueError
from ocp_resources.cluster_service_version import ClusterServiceVersion
from ocp_resources.deployment import Deployment
from ocp_resources.maria_db import MariaDB
from ocp_resources.mariadb_operator import MariadbOperator
from ocp_resources.pod import Pod
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutSampler

from utilities.constants import Timeout

LOGGER = get_logger(name=__name__)


def get_cluster_service_version(client: DynamicClient, prefix: str, namespace: str) -> ClusterServiceVersion:
    csvs = ClusterServiceVersion.get(dyn_client=client, namespace=namespace)

    matching_csvs = [csv for csv in csvs if csv.name.startswith(prefix)]

    if not matching_csvs:
        raise ResourceNotFoundError(f"No ClusterServiceVersion found starting with prefix '{prefix}'")

    if len(matching_csvs) > 1:
        raise ResourceNotUniqueError(
            f"Multiple ClusterServiceVersions found"
            f" starting with prefix '{prefix}':"
            f" {[csv.name for csv in matching_csvs]}"
        )

    return matching_csvs[0]


def wait_for_mariadb_operator_deployments(mariadb_operator: MariadbOperator) -> None:
    expected_deployment_names: list[str] = [
        "mariadb-operator",
        "mariadb-operator-cert-controller",
        "mariadb-operator-helm-controller-manager",
        "mariadb-operator-webhook",
    ]

    for name in expected_deployment_names:
        deployment = Deployment(name=name, namespace=mariadb_operator.namespace)
        deployment.wait_for_replicas()


def wait_for_mariadb_pods(client: DynamicClient, mariadb: MariaDB, timeout: int = Timeout.TIMEOUT_5MIN) -> None:
    def _get_mariadb_pods() -> list[Pod]:
        pods = [
            pod
            for pod in Pod.get(
                dyn_client=client,
                namespace=mariadb.namespace,
                label_selector="app.kubernetes.io/instance=mariadb",
            )
        ]
        return pods

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
