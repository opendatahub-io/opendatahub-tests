"""Utility functions for Spark upgrade tests."""

import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.network_policy import NetworkPolicy
from ocp_resources.pod import Pod
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.service_account import ServiceAccount
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.spark.image_constants import SparkImages
from utilities.resources.spark_application import SparkApplication

LOGGER = structlog.get_logger(name=__name__)

SPARK_VERSION = "4.0.1"
SPARK_IMAGE = SparkImages.DATA_PROCESSING
SPARK_WORKLOAD_SERVICE_ACCOUNT_NAME = "spark-operator-spark"
SPARK_WORKLOAD_ROLE_NAME: str = "spark-role"
SPARK_WORKLOAD_ROLE_BINDING_NAME: str = "spark-role-binding"


def spark_running_execution(client: DynamicClient, spark_app: SparkApplication) -> dict:
    """Capture identity and restart counts only after an executor starts the gated task.

    Args:
        client: Kubernetes client.
        spark_app: Application expected to be running across the upgrade.

    Returns:
        Application and pod identity suitable for comparison after upgrade.
    """
    instance = spark_app.instance
    status = instance.status or {}
    application_state = status.get("applicationState", {})
    if application_state.get("state") in {"FAILED", "SUBMISSION_FAILED"}:
        driver_name = status.get("driverInfo", {}).get("podName")
        driver_logs = "Driver pod has not been reported"
        if driver_name:
            try:
                driver_logs = Pod(client=client, name=driver_name, namespace=spark_app.namespace).log(tail_lines=200)
            except Exception as error:  # noqa: BLE001
                driver_logs = f"Unable to retrieve driver logs: {error}"
        raise RuntimeError(
            f"SparkApplication {spark_app.name} failed: {application_state}. Driver logs:\n{driver_logs}"
        )
    assert application_state.get("state") == "RUNNING", f"Expected an in-progress execution: {instance.status}"
    pods = list(
        Pod.get(
            dyn_client=client,
            namespace=spark_app.namespace,
            label_selector=f"sparkoperator.k8s.io/app-name={spark_app.name}",
        )
    )
    identities = {}
    roles = []
    for pod in pods:
        pod_instance = pod.instance
        role = pod_instance.metadata.labels.get("spark-role")
        roles.append(role)
        assert pod_instance.status.phase == "Running", f"Pod {pod.name} is not running"
        if role == "executor":
            assert "UPGRADE_TASK_STARTED" in pod.log(), "The executor has not started the task"
        identities[pod.name] = {
            "uid": pod_instance.metadata.uid,
            "restarts": sum(item.get("restartCount", 0) for item in pod_instance.status.get("containerStatuses", [])),
        }
    assert sorted(roles) == ["driver", "executor"], f"Expected one driver and one executor, got {roles}"
    return {
        "uid": instance.metadata.uid,
        "generation": instance.metadata.generation,
        "pods": identities,
    }


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


def resubmit_spark_application(
    client: DynamicClient,
    spark_app: SparkApplication,
    teardown: bool,
) -> SparkApplication:
    """Re-run an existing SparkApplication by re-submitting its captured spec.

    Represents the customer scenario of re-running a pre-upgrade workload on the
    upgraded operator. A spark-pi SparkApplication uses restartPolicy: Never, so
    once it reaches COMPLETED it does not run again on its own. To actively *use*
    the existing resource, its spec is captured, the terminal resource is deleted,
    and an identically named/spec'd resource is recreated so the upgraded operator
    reconciles and runs it again.

    Args:
        client: Kubernetes client
        spark_app: The existing (pre-upgrade) SparkApplication to re-run
        teardown: Whether to clean up the recreated resource on teardown

    Returns:
        SparkApplication: The re-submitted SparkApplication resource
    """
    assert spark_app.exists, f"Pre-upgrade SparkApplication {spark_app.name} is missing"
    instance = spark_app.instance
    state = (instance.status or {}).get("applicationState", {}).get("state")
    assert state == "COMPLETED", f"Expected a completed pre-upgrade application before resubmission, got {state}"
    name = spark_app.name
    namespace = spark_app.namespace
    spec = instance.to_dict()["spec"]

    LOGGER.info(f"Re-running existing SparkApplication {name} in namespace {namespace}")

    # Delete the completed resource and wait for it to be fully removed before recreating,
    # so the recreate does not collide with the terminal instance.
    spark_app.clean_up(wait=True)

    kind_dict = {
        "apiVersion": "sparkoperator.k8s.io/v1beta2",
        "kind": "SparkApplication",
        "metadata": {
            "name": name,
            "namespace": namespace,
        },
        "spec": spec,
    }

    resubmitted = SparkApplication(client=client, kind_dict=kind_dict, teardown=teardown)
    resubmitted.deploy()
    LOGGER.info(f"Re-submitted SparkApplication {name} in namespace {namespace}")
    return resubmitted


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
