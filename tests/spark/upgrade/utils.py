"""Utility functions for Spark upgrade tests."""

import structlog
import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.pod import Pod
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from utilities.resources.spark_application import SparkApplication

LOGGER = structlog.get_logger(name=__name__)

UPGRADE_BASELINE_CONFIGMAP = "spark-upgrade-baseline"
SPARK_VERSION = "4.0.1"
SPARK_IMAGE = f"quay.io/opendatahub/data-processing:Spark-v{SPARK_VERSION}"


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
    spark_version: str = SPARK_VERSION,
    image: str = SPARK_IMAGE,
    service_account: str = "spark-operator-spark",
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


def capture_spark_application_baseline(
    client: DynamicClient,
    spark_app: SparkApplication,
) -> dict:
    """Capture baseline state for a SparkApplication.

    Args:
        client: Kubernetes client
        spark_app: SparkApplication resource

    Returns:
        dict: Baseline data including generation and pod restart counts
    """
    LOGGER.info(f"Capturing baseline for SparkApplication {spark_app.name}")

    # Wait for application to complete and get metadata generation
    wait_for_spark_application_state(spark_app=spark_app, expected_state="COMPLETED", timeout=300)
    generation = spark_app.instance.metadata.generation

    # Get pod restart counts
    pod_restart_counts = {}
    pods = list(
        Pod.get(
            dyn_client=client,
            namespace=spark_app.namespace,
            label_selector=f"sparkoperator.k8s.io/app-name={spark_app.name}",
        )
    )

    for pod in pods:
        restart_count = sum(
            container_status.get("restartCount", 0)
            for container_status in pod.instance.status.get("containerStatuses", [])
        )
        pod_restart_counts[pod.name] = restart_count

    baseline = {
        "spark_app_name": spark_app.name,
        "generation": generation,
        "pod_restart_counts": pod_restart_counts,
        "application_state": spark_app.instance.status.get("applicationState", {}).get("state"),
    }

    LOGGER.info(f"Baseline captured: {baseline}")
    return baseline


def save_baseline_to_configmap(
    client: DynamicClient,
    namespace: str,
    baselines: dict,
) -> None:
    """Save baseline data to a ConfigMap.

    Args:
        client: Kubernetes client
        namespace: Namespace containing the ConfigMap
        baselines: Dictionary of baseline data keyed by resource name
    """
    LOGGER.info(f"Saving baseline to ConfigMap {UPGRADE_BASELINE_CONFIGMAP} in namespace {namespace}")

    cm_data = {
        "baselines.yaml": yaml.dump(baselines),
    }

    cm = ConfigMap(
        client=client,
        name=UPGRADE_BASELINE_CONFIGMAP,
        namespace=namespace,
        data=cm_data,
    )

    if cm.exists:
        raise AssertionError(
            f"ConfigMap {UPGRADE_BASELINE_CONFIGMAP} already exists in namespace {namespace}. "
            "This indicates a previous test run did not clean up properly."
        )

    cm.deploy()
    LOGGER.info("Baseline saved to ConfigMap")


def load_baseline_from_configmap(
    client: DynamicClient,
    namespace: str,
) -> dict:
    """Load baseline data from ConfigMap.

    Args:
        client: Kubernetes client
        namespace: Namespace containing the ConfigMap

    Returns:
        dict: Baseline data keyed by resource name
    """
    LOGGER.info(f"Loading baseline from ConfigMap {UPGRADE_BASELINE_CONFIGMAP} in namespace {namespace}")

    cm = ConfigMap(
        client=client,
        name=UPGRADE_BASELINE_CONFIGMAP,
        namespace=namespace,
    )

    if not cm.exists:
        raise AssertionError(
            f"Baseline ConfigMap {UPGRADE_BASELINE_CONFIGMAP} does not exist in namespace {namespace}. "
            "Cannot load baseline for post-upgrade verification."
        )

    baseline_yaml = cm.instance.data.get("baselines.yaml", "")
    baselines = yaml.safe_load(baseline_yaml) or {}

    LOGGER.info(f"Loaded {len(baselines)} baseline entries")
    return baselines


def get_spark_app_baseline(baselines: dict, spark_app_name: str) -> dict:
    """Get baseline for a specific SparkApplication.

    Args:
        baselines: Dictionary of all baselines
        spark_app_name: Name of the SparkApplication

    Returns:
        dict: Baseline data for the SparkApplication
    """
    baseline = baselines.get(spark_app_name)
    if not baseline:
        raise ValueError(f"No baseline found for SparkApplication {spark_app_name}")
    return baseline


def verify_spark_app_generation(
    spark_app: SparkApplication,
    expected_generation: int,
) -> None:
    """Verify SparkApplication metadata generation has not changed.

    Args:
        spark_app: SparkApplication resource
        expected_generation: Expected metadata generation

    Raises:
        AssertionError: If generation doesn't match

    Note:
        Uses metadata.generation (set by Kubernetes) instead of status.observedGeneration
        because Spark Operator doesn't populate observedGeneration in the status.
    """
    actual_generation = spark_app.instance.metadata.generation

    assert actual_generation == expected_generation, (
        f"SparkApplication {spark_app.name} generation changed during upgrade. "
        f"Expected: {expected_generation}, Actual: {actual_generation}"
    )
    LOGGER.info(f"SparkApplication {spark_app.name} generation verified: {actual_generation}")


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


def verify_pods_not_restarted(
    client: DynamicClient,
    spark_app: SparkApplication,
    baseline_restart_counts: dict,
) -> None:
    """Verify pods have not restarted beyond baseline.

    Args:
        client: Kubernetes client
        spark_app: SparkApplication resource
        baseline_restart_counts: Baseline restart counts per pod

    Raises:
        AssertionError: If any pod has restarted beyond baseline
    """
    pods = list(
        Pod.get(
            dyn_client=client,
            namespace=spark_app.namespace,
            label_selector=f"sparkoperator.k8s.io/app-name={spark_app.name}",
        )
    )

    # Verify pod identity continuity
    baseline_pods = set(baseline_restart_counts.keys())
    current_pods = {pod.name for pod in pods}
    assert current_pods == baseline_pods, (
        f"Pod set changed during upgrade. Baseline: {sorted(baseline_pods)}, Current: {sorted(current_pods)}"
    )

    for pod in pods:
        current_restart_count = sum(
            container_status.get("restartCount", 0)
            for container_status in pod.instance.status.get("containerStatuses", [])
        )

        baseline_count = baseline_restart_counts[pod.name]

        assert current_restart_count <= baseline_count, (
            f"Pod {pod.name} restarted during upgrade. Baseline: {baseline_count}, Current: {current_restart_count}"
        )

    LOGGER.info(f"All pods for SparkApplication {spark_app.name} have not restarted beyond baseline")
