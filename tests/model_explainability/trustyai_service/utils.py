import re
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.maria_db import MariaDB
from ocp_resources.mariadb_operator import MariadbOperator
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.trustyai_service import TrustyAIService
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutSampler, retry

from utilities.constants import TRUSTYAI_SERVICE_NAME, Timeout
from utilities.exceptions import TooManyPodsError, UnexpectedFailureError
from utilities.general import validate_container_images, wait_for_pods_by_labels

LOGGER = get_logger(name=__name__)


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


def wait_for_mariadb_pods(client: DynamicClient, mariadb: MariaDB, timeout: int = Timeout.TIMEOUT_15MIN) -> None:
    def _get_mariadb_pods() -> list[Pod]:
        return [
            _pod
            for _pod in Pod.get(
                client=client,
                namespace=mariadb.namespace,
                label_selector=f"app.kubernetes.io/instance={mariadb.name}",
            )
        ]

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
    exceptions_dict={TooManyPodsError: [], UnexpectedFailureError: []},
)
def validate_trustyai_service_db_conn_failure(
    client: DynamicClient, namespace: Namespace, label_selector: str | None
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
    pods = list(Pod.get(client=client, namespace=namespace.name, label_selector=label_selector))
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


@contextmanager
def create_trustyai_service(
    client: DynamicClient,
    namespace: str,
    storage: dict[str, str],
    metrics: dict[str, str],
    name: str = TRUSTYAI_SERVICE_NAME,
    data: dict[str, str] | None = None,
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


@contextmanager
def create_isvc_getter_service_account(
    client: DynamicClient, namespace: Namespace, name: str
) -> Generator[ServiceAccount, Any, Any]:
    """Creates a ServiceAccount for fetching InferenceServices.

    Args:
        client: The Kubernetes dynamic client.
        namespace: Namespace: The Namespace object where the ServiceAccount will be created.
        name: str: The name of the ServiceAccount.

    Yields:
        Generator[ServiceAccount, Any, Any]: The created ServiceAccount object.
    """
    with ServiceAccount(client=client, name=name, namespace=namespace.name) as sa:
        yield sa


@contextmanager
def create_isvc_getter_role(client: DynamicClient, namespace: Namespace, name: str) -> Generator[Role, Any, Any]:
    """Creates a Role with permissions to get, list, and watch InferenceServices.

    Args:
        client: DynamicClient: The Kubernetes dynamic client.
        namespace: Namespace: The Namespace object where the Role will be created.
        name: str: The name of the Role.

    Yields:
        Generator[Role, Any, Any]: The created Role object.
    """
    with Role(
        client=client,
        name=name,
        namespace=namespace.name,
        rules=[
            {
                "apiGroups": ["serving.kserve.io"],
                "resources": ["inferenceservices"],
                "verbs": ["get", "list", "watch"],
            }
        ],
    ) as role:
        yield role


@contextmanager
def create_isvc_getter_role_binding(
    client: DynamicClient, namespace: Namespace, role: Role, service_account: ServiceAccount, name: str
) -> Generator[RoleBinding, Any, Any]:
    """Creates a RoleBinding to link a ServiceAccount to the InferenceService getter Role.

    Args:
        client: DynamicClient: The Kubernetes dynamic client.
        namespace: Namespace: The Namespace object where the RoleBinding will be created.
        role: Role: The Role object to bind.
        service_account: ServiceAccount: The ServiceAccount object to bind.
        name: str: The name of the RoleBinding.

    Yields:
        Generator[RoleBinding, Any, Any]: The created RoleBinding object.
    """
    with RoleBinding(
        client=client,
        name=name,
        namespace=namespace.name,
        subjects_kind="ServiceAccount",
        subjects_name=service_account.name,
        role_ref_kind="Role",
        role_ref_name=role.name,
    ) as rb:
        yield rb


@contextmanager
def create_isvc_getter_token_secret(
    client: DynamicClient, namespace: Namespace, service_account: ServiceAccount, name: str
) -> Generator[Secret, Any, Any]:
    """Creates a Secret of type 'kubernetes.io/service-account-token' for a given ServiceAccount.

    Args:
        client: DynamicClient: The Kubernetes dynamic client.
        namespace: Namespace: The Namespace object where the Secret will be created.
        service_account: ServiceAccount: The ServiceAccount object for which the token Secret is created.
        name: str: The name of the Secret.

    Yields:
        Generator[Secret, Any, Any]: The created Secret object.
    """
    with Secret(
        client=client,
        namespace=namespace.name,
        name=name,
        annotations={"kubernetes.io/service-account.name": service_account.name},
        type="kubernetes.io/service-account-token",
    ) as secret:
        yield secret


def validate_trustyai_service_images(
    client: DynamicClient,
    related_images_refs: set[str],
    model_namespace: Namespace,
    label_selector: str,
    trustyai_operator_configmap: ConfigMap,
) -> None:
    """Validates trustyai service images against a set of related images.

    Args:
        client: DynamicClient: The Kubernetes dynamic client.
        related_images_refs: list[str]: Related images references from RHOAI CSV.
        model_namespace: Namespace: namespace to run the test against.
        label_selector: str: Label selector string to get the trustyai pod.
        trustyai_operator_configmap: ConfigMap: The trustyai operator configmap.

    Returns:
        None

    Raises:
        AssertionError: If any of the related images references are not present or invalid.
    """
    tai_image_refs = {
        value
        for key, value in trustyai_operator_configmap.instance.data.items()
        if key in ["kube-rbac-proxy", "trustyaiServiceImage"]
    }
    trustyai_service_pod = wait_for_pods_by_labels(
        admin_client=client, namespace=model_namespace.name, label_selector=label_selector, expected_num_pods=1
    )[0]
    validation_errors = validate_container_images(pod=trustyai_service_pod, valid_image_refs=tai_image_refs)
    assert len(validation_errors) == 0, validation_errors
    assert tai_image_refs.issubset(related_images_refs), "TrustyAI service container images are not present in CSV."
