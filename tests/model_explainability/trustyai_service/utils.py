from typing import Generator, Any, List, Callable, Optional

from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError, ResourceNotUniqueError
from ocp_resources.cluster_service_version import ClusterServiceVersion
from ocp_resources.deployment import Deployment
from ocp_resources.mariadb_operator import MariadbOperator
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.trustyai_service import TrustyAIService
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutSampler

from tests.model_explainability.trustyai_service.trustyai_service_utils import TRUSTYAI_SERVICE_NAME
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


def get_pods_func(client: DynamicClient, namespace: str, label_selector: Optional[str]) -> Callable[[], list[Pod]]:
    def _get_pods() -> List[Pod]:
        pods = [
            pod
            for pod in Pod.get(
                dyn_client=client,
                namespace=namespace,
                label_selector=label_selector,
            )
        ]
        return pods

    return _get_pods


def wait_for_pods(
    client: DynamicClient,
    namespace: str,
    label_selector: Optional[str] = None,
    timeout: int = Timeout.TIMEOUT_5MIN,
    wait_for_condition: str = Pod.Condition.READY,
    condition_status: str = Pod.Condition.Status.TRUE,
) -> List[Pod]:
    get_pods = get_pods_func(client=client, namespace=namespace, label_selector=label_selector)
    sampler = TimeoutSampler(wait_timeout=timeout, sleep=1, func=lambda: bool(get_pods()))

    for sample in sampler:
        if sample:
            break

    pods = get_pods()
    for pod in pods:
        pod.wait_for_condition(
            condition=wait_for_condition,
            status=condition_status,
            timeout=timeout,  # IDEA: using two different timeouts might be worth considering
        )
    return pods


def wait_for_trustyai_container_terminal_state(
    client: DynamicClient, namespace: str, label_selector: Optional[str] = None, timeout: int = Timeout.TIMEOUT_5MIN
) -> Any | None:
    get_pods = get_pods_func(client, namespace=namespace, label_selector=label_selector)
    sampler = TimeoutSampler(wait_timeout=timeout, sleep=1, func=get_pods)
    for pods in sampler:
        for pod in pods:
            for container_status in pod.instance.status.containerStatuses:
                if (terminate_state := container_status.lastState.terminated) and terminate_state.reason in (
                    pod.Status.ERROR,
                    pod.Status.CRASH_LOOPBACK_OFF,
                ):
                    return terminate_state
    return None


def create_trustyai_db_ca_secret(
    client: DynamicClient, mariadb_ca_cert: str, namespace: Namespace
) -> Generator[None, Any, None]:
    with Secret(
        client=client,
        name=f"{TRUSTYAI_SERVICE_NAME}-db-ca",
        namespace=namespace.name,
        data_dict={"ca.crt": mariadb_ca_cert},
    ):
        yield


def create_trustyai_service(
    client: DynamicClient,
    namespace: Namespace,
    storage: dict[Any, Any],
    metrics: dict[Any, Any],
    data: Optional[dict[Any, Any]] = None,
    wait_for_replicas: bool = True,
) -> Generator[TrustyAIService, Any, Any]:
    with TrustyAIService(
        client=client,
        name=TRUSTYAI_SERVICE_NAME,
        namespace=namespace.name,
        storage=storage,
        metrics=metrics,
        data=data,
    ) as trustyai_service:
        trustyai_deployment = Deployment(namespace=namespace, name=TRUSTYAI_SERVICE_NAME, wait_for_resource=True)
        if wait_for_replicas:
            trustyai_deployment.wait_for_replicas()
        yield trustyai_service
