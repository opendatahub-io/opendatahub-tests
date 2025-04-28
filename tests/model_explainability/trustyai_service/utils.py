from typing import Generator, Any, Optional
import re

from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError, ResourceNotUniqueError
from ocp_resources.cluster_service_version import ClusterServiceVersion
from ocp_resources.deployment import Deployment
from ocp_resources.mariadb_operator import MariadbOperator
from ocp_resources.maria_db import MariaDB
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.trustyai_service import TrustyAIService
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutSampler
from tests.model_explainability.trustyai_service.trustyai_service_utils import TRUSTYAI_SERVICE_NAME
from utilities.constants import Timeout
from timeout_sampler import retry

from utilities.exceptions import TooManyPodsError, UnexpectedFailureError

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
        _pods = [
            _pod
            for _pod in Pod.get(
                dyn_client=client,
                namespace=mariadb.namespace,
                label_selector="app.kubernetes.io/instance=mariadb",
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


@retry(
    wait_timeout=Timeout.TIMEOUT_2MIN,
    sleep=5,
    exceptions_dict={TooManyPodsError: list(), UnexpectedFailureError: list()},
)
def validate_trustyai_service_db_conn_failure(
    client: DynamicClient, namespace: Namespace, label_selector: Optional[str]
) -> bool:
    """Validate if invalid DB Certificate leads to pod crash loop.

    Waits for TrustyAIService pod to fail and checks if the pod is in a CrashLoopBackOff state and
    the LastState is in terminated state and the cause was a MariaDB TLS certificate exception.
    Also checks if there are more than one pod for the service.

    Args:
        client: The OpenShift client.
        namespace: Namespace under which the pod is created.
        label_selector: The label selector used to select the correct pod(s) to monitor.

    Returns:
        bool: True if pod failure is of expected state else False.

    Raises:
        TimeoutExpiredError: if the method takes longer than `wait_timeout` to return a value.
        TooManyPodsError: if the number of pods exceeds 1.
        UnexpectedFailureError: if the pod failure is different from the expected failure mode.

    """
    pods = list(Pod.get(dyn_client=client, namespace=namespace.name, label_selector=label_selector))
    mariadb_conn_failure_regex = (
        r"^.+ERROR.+Could not connect to mariadb:.+ PKIX path validation failed: "
        r"java\.security\.cert\.CertPathValidatorException: signature check failed"
    )
    if pods:
        if len(pods) > 1:
            raise TooManyPodsError("More than one pod found in TrustyAIService.")
        for container_status in pods[0].instance.status.containerStatuses:
            if (terminate_state := container_status.lastState.terminated) and terminate_state.reason in (
                pods[0].Status.ERROR,
                pods[0].Status.CRASH_LOOPBACK_OFF,
            ):
                if not re.search(mariadb_conn_failure_regex, terminate_state.message):
                    raise UnexpectedFailureError(
                        f"Service {TRUSTYAI_SERVICE_NAME} did not fail with a mariadb connection failure as expected.\
                                  \nExpected format: {mariadb_conn_failure_regex}\
                                  \nGot: {terminate_state.message}"
                    )
                return True
    return False


def create_trustyai_service(
    client: DynamicClient,
    namespace: str,
    storage: dict[str, str],
    metrics: dict[str, str],
    name: str = TRUSTYAI_SERVICE_NAME,
    data: Optional[dict[str, str]] = None,
    wait_for_replicas: bool = True,
    teardown: bool = True,
) -> Generator[TrustyAIService, Any, Any]:
    """Creates TrustyAIService and TrustyAI deployment.

    Args:
         client: the client.
         namespace: Namespace to create the service in.
         storage: Dict with storage configuration.
         metrics: Dict with metrics configuration.
         name: Name of the TrustyAI service and deployment (default "trustyai-service").
         data: An optional dict with data.
         wait_for_replicas: Wait until replicas are available (default True).
         teardown: Teardown the service (default True).

    Yields:
         Generator[TrustyAIService, Any, Any]: The TrustyAI service.
    """
    with TrustyAIService(
        client=client,
        name=name,
        namespace=namespace,
        storage=storage,
        metrics=metrics,
        data=data,
        teardown=teardown,
    ) as trustyai_service:
        trustyai_deployment = Deployment(namespace=namespace, name=name, wait_for_resource=True)
        if wait_for_replicas:
            trustyai_deployment.wait_for_replicas()
        yield trustyai_service
