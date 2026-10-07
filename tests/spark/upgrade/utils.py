"""Utilities for Spark execution continuity and resubmission across upgrades."""

import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.pod import Pod

from utilities.resources.spark_application import SparkApplication

LOGGER = structlog.get_logger(name=__name__)


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
