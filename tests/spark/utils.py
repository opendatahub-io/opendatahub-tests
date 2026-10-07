"""Shared workload utilities for Spark install and upgrade tests."""

import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.network_policy import NetworkPolicy
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.service_account import ServiceAccount
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.spark.image_constants import SparkImages
from utilities.resources.spark_application import SparkApplication

LOGGER = structlog.get_logger(name=__name__)

UPGRADE_NAMESPACE = "upgrade-spark-operator"

SPARK_VERSION = "4.0.1"
SPARK_IMAGE = SparkImages.DATA_PROCESSING
SPARK_WORKLOAD_SERVICE_ACCOUNT_NAME = "spark-operator-spark"
SPARK_WORKLOAD_ROLE_NAME: str = "spark-role"
SPARK_WORKLOAD_ROLE_BINDING_NAME: str = "spark-role-binding"


def wait_for_spark_application_state(
    spark_app: SparkApplication,
    expected_state: str,
    timeout: int = 300,
) -> None:
    """Wait for SparkApplication to reach expected state.

    Args:
        spark_app: SparkApplication resource
        expected_state: Expected application state (e.g., "COMPLETED", "RUNNING")
        timeout: Timeout in seconds

    Raises:
        TimeoutExpiredError: If the state is not reached within timeout
    """
    LOGGER.info(f"Waiting for SparkApplication {spark_app.name} to reach state {expected_state}")

    def _get_state():
        """Get application state, handling None status."""
        status = spark_app.instance.status
        if status is None:
            return None
        return status.get("applicationState", {}).get("state")

    sampler = TimeoutSampler(
        wait_timeout=timeout,
        sleep=5,
        func=_get_state,
    )

    try:
        for state in sampler:
            if state == expected_state:
                LOGGER.info(f"SparkApplication {spark_app.name} reached state {expected_state}")
                return
    except TimeoutExpiredError:
        status = spark_app.instance.status
        if status is None:
            current_state = "No status yet"
        else:
            current_state = status.get("applicationState", {}).get("state", "UNKNOWN")
        raise TimeoutExpiredError(
            f"SparkApplication {spark_app.name} did not reach {expected_state} state within {timeout}s. "
            f"Current state: {current_state}"
        )


def create_spark_pi_application_spec(
    name: str,
    namespace: str,
    service_account: str,
    spark_version: str = SPARK_VERSION,
    image: str = SPARK_IMAGE,
) -> dict:
    """Create a SparkApplication spec for spark-pi workload.

    Args:
        name: Name of the SparkApplication
        namespace: Namespace to deploy to
        spark_version: Spark version (default: 4.0.1)
        image: Spark image to use
        service_account: Service account for Spark pods

    Returns:
        dict: SparkApplication spec matching the Go implementation
    """
    return {
        "apiVersion": "sparkoperator.k8s.io/v1beta2",
        "kind": "SparkApplication",
        "metadata": {
            "name": name,
            "namespace": namespace,
        },
        "spec": {
            "type": "Scala",
            "mode": "cluster",
            "image": image,
            "imagePullPolicy": "IfNotPresent",
            "mainClass": "org.apache.spark.examples.SparkPi",
            "mainApplicationFile": f"local:///opt/spark/examples/jars/spark-examples_2.13-{spark_version}.jar",
            "sparkVersion": spark_version,
            "restartPolicy": {
                "type": "Never",
            },
            "driver": {
                "cores": 1,
                "memory": "512m",
                "serviceAccount": service_account,
                "volumeMounts": [
                    {
                        "name": "work-dir",
                        "mountPath": "/opt/spark/work-dir",
                    }
                ],
            },
            "executor": {
                "cores": 1,
                "instances": 1,
                "memory": "512m",
                "volumeMounts": [
                    {
                        "name": "work-dir",
                        "mountPath": "/opt/spark/work-dir",
                    }
                ],
            },
            "volumes": [
                {
                    "name": "work-dir",
                    "emptyDir": {},
                }
            ],
        },
    }


def verify_spark_app_completed(spark_app: SparkApplication) -> None:
    """Verify SparkApplication reached COMPLETED state.

    Args:
        spark_app: SparkApplication resource

    Raises:
        AssertionError: If application is not in COMPLETED state
    """
    wait_for_spark_application_state(spark_app=spark_app, expected_state="COMPLETED", timeout=300)
    state = spark_app.instance.status.get("applicationState", {}).get("state")
    assert state == "COMPLETED", f"SparkApplication {spark_app.name} not in COMPLETED state. Actual: {state}"
    LOGGER.info(f"SparkApplication {spark_app.name} is in COMPLETED state")


def get_spark_network_policies(client: DynamicClient, namespace: str) -> list[NetworkPolicy]:
    """Find only the Spark internal-traffic NetworkPolicy.

    Args:
        client: Kubernetes client.
        namespace: Namespace to search.

    Returns:
        The matching NetworkPolicy, or an empty list if it is absent.
    """
    return [
        network_policy
        for network_policy in NetworkPolicy.get(client=client, namespace=namespace)
        if network_policy.name == "spark-operator-allow-internal"
    ]


def get_spark_roles(client: DynamicClient, namespace: str) -> list[Role]:
    """Find only the named Spark workload Role.

    Args:
        client: Kubernetes client.
        namespace: Namespace to search.

    Returns:
        The matching workload Role, or an empty list if it is absent.
    """
    return [role for role in Role.get(client=client, namespace=namespace) if role.name == SPARK_WORKLOAD_ROLE_NAME]


def get_spark_service_accounts(client: DynamicClient, namespace: str) -> list[ServiceAccount]:
    """Find only the named Spark workload ServiceAccount.

    Args:
        client: Kubernetes client.
        namespace: Namespace to search.

    Returns:
        The matching workload ServiceAccount, or an empty list if it is absent.
    """
    return [
        service_account
        for service_account in ServiceAccount.get(client=client, namespace=namespace)
        if service_account.name == SPARK_WORKLOAD_SERVICE_ACCOUNT_NAME
    ]


def get_spark_role_bindings(client: DynamicClient, namespace: str) -> list[RoleBinding]:
    """Find only the named Spark workload RoleBinding.

    Args:
        client: Kubernetes client.
        namespace: Namespace to search.

    Returns:
        The matching workload RoleBinding, or an empty list if it is absent.
    """
    return [
        role_binding
        for role_binding in RoleBinding.get(client=client, namespace=namespace)
        if role_binding.name == SPARK_WORKLOAD_ROLE_BINDING_NAME
    ]


def recreate_role_in_namespace(
    client: DynamicClient,
    source_role: Role,
    target_namespace: str,
    teardown: bool,
) -> Role:
    source_dict = source_role.instance.to_dict()
    source_dict["metadata"] = {"name": source_role.name, "namespace": target_namespace}
    role = Role(client=client, kind_dict=source_dict, rules=source_dict["rules"], teardown=teardown)
    role.deploy()
    return role


def recreate_service_account_in_namespace(
    client: DynamicClient,
    source_sa: ServiceAccount,
    target_namespace: str,
    teardown: bool,
) -> ServiceAccount:
    source_dict = source_sa.instance.to_dict()
    source_dict["metadata"] = {"name": source_sa.name, "namespace": target_namespace}
    # secrets are auto-managed by the API server and must not be copied from the source
    source_dict.pop("secrets", None)
    sa = ServiceAccount(client=client, kind_dict=source_dict, teardown=teardown)
    sa.deploy()
    return sa


def recreate_role_binding_in_namespace(
    client: DynamicClient,
    source_rb: RoleBinding,
    target_namespace: str,
    teardown: bool,
) -> RoleBinding:
    source_dict = source_rb.instance.to_dict()
    source_dict["metadata"] = {"name": source_rb.name, "namespace": target_namespace}
    for subject in source_dict.get("subjects", []):
        subject["namespace"] = target_namespace
    rb = RoleBinding(client=client, kind_dict=source_dict, teardown=teardown)
    rb.deploy()
    return rb


def recreate_network_policy_in_namespace(
    client: DynamicClient,
    source_np: NetworkPolicy,
    target_namespace: str,
    teardown: bool,
) -> NetworkPolicy:
    source_dict = source_np.instance.to_dict()
    source_dict["metadata"] = {"name": source_np.name, "namespace": target_namespace}
    np = NetworkPolicy(client=client, kind_dict=source_dict, teardown=teardown)
    np.deploy()
    return np
