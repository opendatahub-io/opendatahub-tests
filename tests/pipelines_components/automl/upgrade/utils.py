import uuid
from typing import Any

import structlog
import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.pod import Pod
from timeout_sampler import TimeoutExpiredError

from tests.pipelines_components.constants import (
    DSPA_NAME,
    DSPA_S3_BUCKET,
    DSPA_S3_SECRET,
    MINIO_MC_IMAGE,
    MINIO_UPLOADER_SECURITY_CONTEXT,
)
from utilities.general import collect_pod_information

LOGGER = structlog.get_logger(name=__name__)

UPGRADE_BASELINE_CONFIGMAP = "automl-upgrade-baseline"
TIMESERIES_UPGRADE_BASELINE_CONFIGMAP = "automl-timeseries-upgrade-baseline"

REGRESSION_V2_INPUT: dict[str, Any] = {
    "inputs": [
        {"name": "feature_a", "shape": [1], "datatype": "FP64", "data": [1.0]},
        {"name": "feature_b", "shape": [1], "datatype": "FP64", "data": [2.0]},
    ]
}

TIMESERIES_SUNSPOTS_V1_INPUT: dict[str, Any] = {
    "instances": [
        {"item_id": "sunspots", "timestamp": "1749-01", "Sunspots": 58.0},
        {"item_id": "sunspots", "timestamp": "1749-02", "Sunspots": 62.6},
        {"item_id": "sunspots", "timestamp": "1749-03", "Sunspots": 70.0},
        {"item_id": "sunspots", "timestamp": "1749-04", "Sunspots": 55.7},
        {"item_id": "sunspots", "timestamp": "1749-05", "Sunspots": 85.0},
    ]
}


def save_baseline_to_configmap(
    client: DynamicClient,
    namespace: str,
    baselines: dict,
    configmap_name: str = UPGRADE_BASELINE_CONFIGMAP,
) -> None:
    """Save baseline data to a ConfigMap for post-upgrade verification."""
    LOGGER.info(f"Saving baseline to ConfigMap {configmap_name} in namespace {namespace}")

    cm = ConfigMap(
        client=client,
        name=configmap_name,
        namespace=namespace,
        data={"baselines.yaml": yaml.dump(baselines)},
    )

    if cm.exists:
        raise AssertionError(
            f"ConfigMap {configmap_name} already exists in namespace {namespace}. "
            "This indicates a previous test run did not clean up properly."
        )

    cm.deploy()
    LOGGER.info("Baseline saved to ConfigMap")


def load_baseline_from_configmap(
    client: DynamicClient,
    namespace: str,
    configmap_name: str = UPGRADE_BASELINE_CONFIGMAP,
) -> dict:
    """Load baseline data from ConfigMap saved during pre-upgrade."""
    LOGGER.info(f"Loading baseline from ConfigMap {configmap_name} in namespace {namespace}")

    cm = ConfigMap(
        client=client,
        name=configmap_name,
        namespace=namespace,
    )

    if not cm.exists:
        raise AssertionError(
            f"Baseline ConfigMap {configmap_name} does not exist in namespace {namespace}. "
            "Cannot load baseline for post-upgrade verification."
        )

    baseline_yaml = cm.instance.data.get("baselines.yaml", "")
    baselines = yaml.safe_load(baseline_yaml) or {}

    LOGGER.info(f"Loaded baseline: {baselines}")
    return baselines


def discover_model_path(
    admin_client: DynamicClient,
    namespace: str,
    run_id: str,
    pipeline_name: str = "autogluon-tabular-training-pipeline",
) -> str:
    """Find the trained model predictor path in DSPA MinIO after a pipeline run completes.

    Uses an mc pod to list the pipeline artifacts and find the predictor directory.
    Returns an S3 URI like s3://mlpipeline/<pipeline_name>/<run_id>/.../predictor
    """
    minio_endpoint = f"http://minio-{DSPA_NAME}.{namespace}.svc.cluster.local:9000"
    run_prefix = f"{DSPA_S3_BUCKET}/{pipeline_name}/{run_id}/"

    script = (
        "export MC_CONFIG_DIR=/work/.mc && "
        "mc alias set dspa $DST_ENDPOINT $DST_ACCESS_KEY $DST_SECRET_KEY && "
        f"mc ls --recursive dspa/{run_prefix}"
    )

    pod_name = f"automl-model-finder-{uuid.uuid4().hex[:8]}"
    with Pod(
        client=admin_client,
        name=pod_name,
        namespace=namespace,
        restart_policy="Never",
        volumes=[{"name": "work", "emptyDir": {}}],
        containers=[
            {
                "name": "mc-finder",
                "image": MINIO_MC_IMAGE,
                "command": ["/bin/sh", "-c"],
                "args": [script],
                "volumeMounts": [{"name": "work", "mountPath": "/work"}],
                "securityContext": MINIO_UPLOADER_SECURITY_CONTEXT,
                "env": [
                    {"name": "DST_ENDPOINT", "value": minio_endpoint},
                    {
                        "name": "DST_ACCESS_KEY",
                        "valueFrom": {"secretKeyRef": {"name": DSPA_S3_SECRET, "key": "accesskey"}},
                    },
                    {
                        "name": "DST_SECRET_KEY",
                        "valueFrom": {"secretKeyRef": {"name": DSPA_S3_SECRET, "key": "secretkey"}},
                    },
                ],
            }
        ],
        wait_for_resource=True,
    ) as finder_pod:
        try:
            finder_pod.wait_for_status(status="Succeeded", timeout=120)
        except TimeoutExpiredError:
            logs = finder_pod.log(container="mc-finder")
            LOGGER.error(f"Model finder pod failed. Logs:\n{logs}")
            collect_pod_information(pod=finder_pod)
            raise

        logs = finder_pod.log(container="mc-finder")

    LOGGER.info(f"mc ls output for run {run_id}:\n{logs}")

    predictor_paths = set()
    for line in logs.strip().splitlines():
        if "/predictor/" in line:
            # mc ls format: "[date time zone] size class path" — split into at most 6 parts
            # so the path (which may theoretically contain spaces) stays intact in parts[5]
            parts = line.strip().split(None, 5)
            if len(parts) < 6:
                continue
            file_path = parts[5]
            predictor_dir = file_path.split("/predictor/")[0] + "/predictor"
            predictor_paths.add(predictor_dir)

    assert predictor_paths, f"No predictor directory found in MinIO under {run_prefix}. mc ls output:\n{logs}"

    relative_path = sorted(predictor_paths)[0]
    storage_uri = f"s3://{run_prefix}{relative_path.rstrip('/')}/"

    LOGGER.info(f"Discovered model path: {storage_uri}")
    return storage_uri
