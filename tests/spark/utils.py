"""Shared workload utilities for Spark install and upgrade tests."""

import shlex

import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.network_policy import NetworkPolicy
from ocp_resources.pod import Pod
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.service import Service
from ocp_resources.service_account import ServiceAccount
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.spark.image_constants import SparkImages
from utilities.resources.scheduled_spark_application import ScheduledSparkApplication
from utilities.resources.spark_application import SparkApplication
from utilities.resources.spark_connect import SparkConnect

LOGGER = structlog.get_logger(name=__name__)

UPGRADE_NAMESPACE = "upgrade-spark-operator"

SPARK_VERSION = "4.0.1"
SPARK_IMAGE = SparkImages.DATA_PROCESSING
SPARK_WORKLOAD_SERVICE_ACCOUNT_NAME = "spark-operator-spark"
SPARK_WORKLOAD_ROLE_NAME: str = "spark-role"
SPARK_WORKLOAD_ROLE_BINDING_NAME: str = "spark-role-binding"

# gRPC port the Spark Connect server listens on; clients connect to sc://<service>:<port>.
SPARK_CONNECT_PORT: int = 15002
# OpenShift restricted-SCC-compatible container security context (no uid pinning; the SCC assigns one).
RESTRICTED_SECURITY_CONTEXT: dict = {
    "allowPrivilegeEscalation": False,
    "runAsNonRoot": True,
    "capabilities": {"drop": ["ALL"]},
    "seccompProfile": {"type": "RuntimeDefault"},
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


def create_gated_continuity_spark_spec(
    name: str,
    namespace: str,
    service_account: str,
    control_config_map_name: str,
) -> dict:
    """Build a Python SparkApplication spec whose task is gated on a mounted release file.

    The workload polls ``/opt/spark/upgrade/release`` and only completes once a release flag is
    set to ``"true"``, so the execution can be held in-flight across an operator upgrade. The same
    spec is used for a directly-created SparkApplication and as the ``template`` of a
    ScheduledSparkApplication, which keeps both continuity tests on one shared workload definition.

    Args:
        name: Name used to seed the underlying spark-pi base spec.
        namespace: Namespace the application runs in.
        service_account: Driver service account.
        control_config_map_name: ConfigMap carrying ``workload.py`` and the ``release`` gate.

    Returns:
        dict: The SparkApplication ``spec`` with the control ConfigMap mounted into driver and executor.
    """
    spec = create_spark_pi_application_spec(name=name, namespace=namespace, service_account=service_account)["spec"]
    spec.pop("mainClass")
    spec.update({
        "type": "Python",
        "pythonVersion": "3",
        "mainApplicationFile": "local:///opt/spark/upgrade/workload.py",
        "sparkConf": {"spark.task.maxFailures": "1", "spark.speculation": "false"},
    })
    spec["volumes"].append({"name": "upgrade-control", "configMap": {"name": control_config_map_name}})
    volumes = spec.pop("volumes")
    for role in ("driver", "executor"):
        mounts = spec[role].pop("volumeMounts")
        mounts.append({"name": "upgrade-control", "mountPath": "/opt/spark/upgrade", "readOnly": True})
        # Native pod templates carry mounts even when the mutating webhook does not inject them.
        spec[role]["template"] = {
            "spec": {
                "volumes": volumes,
                "containers": [{"name": f"spark-kubernetes-{role}", "volumeMounts": mounts}],
            }
        }
    return spec


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


def scheduled_last_run_application(client: DynamicClient, scheduled_app: ScheduledSparkApplication) -> SparkApplication:
    """Resolve the SparkApplication created for a ScheduledSparkApplication's most recent run.

    Args:
        client: Kubernetes client.
        scheduled_app: ScheduledSparkApplication whose latest child run is wanted.

    Returns:
        SparkApplication: The child named by ``status.lastRunName``.

    Raises:
        AssertionError: If the scheduler has not recorded a run yet.
    """
    last_run_name = (scheduled_app.instance.status or {}).get("lastRunName")
    assert last_run_name, f"ScheduledSparkApplication {scheduled_app.name} has not started a run yet"
    return SparkApplication(client=client, name=last_run_name, namespace=scheduled_app.namespace)


def wait_for_scheduled_spark_application_run(
    client: DynamicClient,
    scheduled_app: ScheduledSparkApplication,
    timeout: int = 300,
) -> SparkApplication:
    """Wait until the scheduler records a run, then return that child SparkApplication.

    Args:
        client: Kubernetes client.
        scheduled_app: ScheduledSparkApplication expected to fire on its schedule.
        timeout: Maximum seconds to wait for the first run to be recorded.

    Returns:
        SparkApplication: The child named by ``status.lastRunName``.

    Raises:
        TimeoutExpiredError: If no run is recorded within the timeout.
    """
    LOGGER.info(f"Waiting for ScheduledSparkApplication {scheduled_app.name} to start a run")
    for status in TimeoutSampler(
        wait_timeout=timeout,
        sleep=10,
        func=lambda: scheduled_app.instance.status or {},
    ):
        if last_run_name := status.get("lastRunName"):
            LOGGER.info(f"ScheduledSparkApplication {scheduled_app.name} started run {last_run_name}")
            return SparkApplication(client=client, name=last_run_name, namespace=scheduled_app.namespace)
    raise TimeoutExpiredError(f"ScheduledSparkApplication {scheduled_app.name} did not start a run within {timeout}s")


def create_spark_connect_spec(
    name: str,
    namespace: str,
    service_account: str,
    executor_instances: int = 1,
    spark_version: str = SPARK_VERSION,
    image: str = SPARK_IMAGE,
) -> dict:
    """Create a SparkConnect spec for a long-lived Spark Connect server.

    Args:
        name: Name of the SparkConnect resource.
        namespace: Namespace to deploy to.
        service_account: Service account for the server (driver) pod.
        executor_instances: Number of executor pods the server runs.
        spark_version: Spark version (default: 4.0.1).
        image: Spark image for the server and executor pods.

    Returns:
        dict: SparkConnect manifest (apiVersion sparkoperator.k8s.io/v1alpha1).
    """
    return {
        "apiVersion": "sparkoperator.k8s.io/v1alpha1",
        "kind": "SparkConnect",
        "metadata": {"name": name, "namespace": namespace},
        "spec": {
            "sparkVersion": spark_version,
            "server": {
                "template": {
                    "spec": {
                        "serviceAccount": service_account,
                        "containers": [
                            {
                                "name": "spark-kubernetes-driver",
                                "image": image,
                                "imagePullPolicy": "IfNotPresent",
                                "securityContext": RESTRICTED_SECURITY_CONTEXT,
                            }
                        ],
                    }
                },
            },
            "executor": {
                "instances": executor_instances,
                "cores": 1,
                "memory": "512m",
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": "spark-kubernetes-executor",
                                "image": image,
                                "imagePullPolicy": "IfNotPresent",
                                "securityContext": RESTRICTED_SECURITY_CONTEXT,
                            }
                        ],
                    }
                },
            },
        },
    }


def wait_for_spark_connect_ready(
    client: DynamicClient,
    spark_connect: SparkConnect,
    executor_instances: int = 1,
    timeout: int = 900,
) -> str:
    """Wait until the server pod, all executors, and the gRPC service are ready.

    Args:
        client: Kubernetes client.
        spark_connect: SparkConnect whose server should become serviceable.
        executor_instances: Expected number of ready executor pods.
        timeout: Maximum seconds to wait.

    Returns:
        str: The server service name (clients connect to ``sc://<service>.<ns>.svc:15002``).

    Raises:
        TimeoutExpiredError: If the server does not become ready within the timeout.
    """
    LOGGER.info(f"Waiting for SparkConnect {spark_connect.name} server to become ready")
    executor_selector = {
        "sparkoperator.k8s.io/launched-by-spark-operator": "true",
        "sparkoperator.k8s.io/connect-name": spark_connect.name,
        "spark-role": "executor",
    }
    for status in TimeoutSampler(wait_timeout=timeout, sleep=15, func=lambda: spark_connect.instance.status or {}):
        server = status.get("server", {})
        server_pod_name, service_name = server.get("podName"), server.get("serviceName")
        if not (server_pod_name and service_name):
            continue

        server_pod = Pod(client=client, name=server_pod_name, namespace=spark_connect.namespace)
        if not (server_pod.exists and server_pod.instance.status.phase == "Running"):
            continue

        executors = list(
            Pod.get(
                dyn_client=client,
                namespace=spark_connect.namespace,
                label_selector=",".join(f"{key}={value}" for key, value in executor_selector.items()),
            )
        )
        # Count only running executors; the selector also matches Failed/Succeeded pods the operator
        # has replaced, which would otherwise inflate the count and never match executor_instances.
        running_executors = [executor for executor in executors if executor.instance.status.phase == "Running"]
        if len(running_executors) != executor_instances:
            continue

        service = Service(client=client, name=service_name, namespace=spark_connect.namespace)
        if not service.exists:
            continue
        if not any(port.port == SPARK_CONNECT_PORT for port in service.instance.spec.ports):
            continue

        LOGGER.info(f"SparkConnect {spark_connect.name} server is ready on service {service_name}")
        return service_name
    raise TimeoutExpiredError(f"SparkConnect {spark_connect.name} server did not become ready within {timeout}s")


def run_spark_connect_query(client_pod: Pod, service_name: str, namespace: str, timeout: int = 300) -> str:
    """Run a Scala range/doubled query against the Spark Connect endpoint from a client pod.

    Uses the Scala Spark Connect client bundled in the Spark distribution (no PySpark/pip needed), so
    the query runs in a disconnected-friendly image. The session is built explicitly rather than with
    ``--remote`` so the classic REPL handles piped, non-interactive input.

    Args:
        client_pod: A running pod with the Spark distribution (``${SPARK_HOME}/bin/spark-shell``).
        service_name: The Spark Connect server service name.
        namespace: Namespace of the server service.
        timeout: Maximum seconds to wait for the query to complete.

    Returns:
        str: Combined stdout of the spark-shell session.
    """
    connect_url = f"sc://{service_name}.{namespace}.svc.cluster.local:{SPARK_CONNECT_PORT}"
    scala_script = (
        "import org.apache.spark.sql.SparkSession\n"
        f'val remoteSpark = SparkSession.builder().remote("{connect_url}").getOrCreate()\n'
        'val df = remoteSpark.range(100).selectExpr("id", "id * 2 as doubled")\n'
        'println("ROW_COUNT=" + df.count())\n'
        'println("VALIDATE=" + df.filter("id = 7").collect()(0).getAs[Long]("doubled"))\n'
        "System.exit(0)\n"
    )
    spark_shell_command = (
        f"export HOME=/tmp && printf '%s' {shlex.quote(scala_script)} | "
        "${SPARK_HOME}/bin/spark-shell --master 'local[1]' --conf spark.ui.enabled=false"
    )
    return client_pod.execute(command=["bash", "-c", spark_shell_command], timeout=timeout)


def verify_spark_connect_query(output: str) -> None:
    """Assert a Spark Connect range/doubled query returned the expected results.

    Args:
        output: stdout from :func:`run_spark_connect_query`.

    Raises:
        AssertionError: If the query did not return 100 rows or the doubled value of 14 for id=7.
    """
    assert "ROW_COUNT=100" in output, f"Expected the query to return 100 rows, got: {output}"
    assert "VALIDATE=14" in output, f"Expected a doubled value of 14 for id=7, got: {output}"


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
