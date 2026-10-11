import os
import shlex
import uuid
from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.data_science_pipelines_application import DataSciencePipelinesApplication
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from timeout_sampler import TimeoutExpiredError

from tests.pipelines_components.automl.data.synthetic import synthetic_automl_csv
from tests.pipelines_components.constants import (
    AUTOML_TASK_CONFIGS,
    AUTOML_TIMESERIES_CONFIG,
    AUTOML_TRAIN_DATA_FILE_KEY,
    DSPA_NAME,
    DSPA_READY_BUFFER_SECONDS,
    DSPA_S3_BUCKET,
    DSPA_S3_SECRET,
    MANAGED_PIPELINE_AUTOML_TABULAR,
    MANAGED_PIPELINE_AUTOML_TIMESERIES,
    MANAGED_PIPELINE_POLL_INTERVAL,
    MANAGED_PIPELINE_WAIT_TIMEOUT,
    MINIO_MC_IMAGE,
    MINIO_UPLOADER_SECURITY_CONTEXT,
)
from tests.pipelines_components.utils import create_pipeline_run_managed, delete_pipeline_run, wait_for_managed_pipeline
from utilities.general import collect_pod_information


@pytest.fixture(scope="function")
def automl_managed_pipeline(
    dspa: DataSciencePipelinesApplication,
    dspa_api_url: str,
    dspa_auth_headers: dict[str, str],
    dspa_ca_bundle_file: str,
    task_type: str,
) -> dict[str, str]:
    """Discover the managed pipeline for the selected task type."""
    display_name = MANAGED_PIPELINE_AUTOML_TIMESERIES if task_type == "timeseries" else MANAGED_PIPELINE_AUTOML_TABULAR
    return wait_for_managed_pipeline(
        api_url=dspa_api_url,
        headers=dspa_auth_headers,
        display_name=display_name,
        ca_bundle=dspa_ca_bundle_file,
        timeout=DSPA_READY_BUFFER_SECONDS + MANAGED_PIPELINE_WAIT_TIMEOUT,
        poll_interval=MANAGED_PIPELINE_POLL_INTERVAL,
    )


@pytest.fixture(scope="function")
def automl_train_data(
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
    dspa_s3_credentials: Secret,
    task_type: str,
) -> str:
    """Generate 150 training rows and upload their CSV to DSPA MinIO."""
    destination_key = f"{task_type}/{AUTOML_TRAIN_DATA_FILE_KEY}"
    minio_endpoint = f"http://minio-{DSPA_NAME}.{pipelines_namespace.name}.svc.cluster.local:9000"
    bucket = shlex.quote(s=DSPA_S3_BUCKET)
    key = shlex.quote(s=destination_key)
    script = (
        "export MC_CONFIG_DIR=/work/.mc && "
        "mc alias set dspa $DST_ENDPOINT $DST_ACCESS_KEY $DST_SECRET_KEY && "
        f"mc mb --ignore-existing dspa/{bucket} && "
        f"mc cp /data/train.csv dspa/{bucket}/{key}"
    )

    with (
        ConfigMap(
            client=admin_client,
            name=f"automl-train-{task_type}-{uuid.uuid4().hex[:8]}",
            namespace=pipelines_namespace.name,
            data={"train.csv": synthetic_automl_csv(task_type=task_type)},
        ) as training_data,
        Pod(
            client=admin_client,
            name=f"automl-data-uploader-{uuid.uuid4().hex[:8]}",
            namespace=pipelines_namespace.name,
            restart_policy="Never",
            volumes=[
                {"name": "work", "emptyDir": {}},
                {"name": "training-data", "configMap": {"name": training_data.name}},
            ],
            containers=[
                {
                    "name": "minio-uploader",
                    "image": MINIO_MC_IMAGE,
                    "command": ["/bin/sh", "-c"],
                    "args": [script],
                    "volumeMounts": [
                        {"name": "work", "mountPath": "/work"},
                        {"name": "training-data", "mountPath": "/data", "readOnly": True},
                    ],
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
        ) as upload_pod,
    ):
        try:
            upload_pod.wait_for_status(status="Succeeded", timeout=120)
        except TimeoutExpiredError:
            collect_pod_information(pod=upload_pod)
            raise

    return destination_key


@pytest.fixture(scope="function")
def automl_run_id(
    dspa_api_url: str,
    dspa_auth_headers: dict[str, str],
    dspa_ca_bundle_file: str,
    automl_managed_pipeline: dict[str, str],
    pipelines_namespace: Namespace,
    automl_train_data: str,
    task_type: str,
) -> Generator[str, Any, Any]:
    """Submit the selected managed AutoML pipeline using its uploaded dataset."""
    parameters: dict[str, Any] = {
        "train_data_secret_name": DSPA_S3_SECRET,
        "train_data_bucket_name": DSPA_S3_BUCKET,
        "train_data_file_key": automl_train_data,
    }
    if task_type == "timeseries":
        parameters.update(AUTOML_TIMESERIES_CONFIG)
    else:
        task_config = AUTOML_TASK_CONFIGS[task_type]
        parameters.update({key: task_config[key] for key in ("label_column", "task_type", "top_n")})

    run_id = create_pipeline_run_managed(
        api_url=dspa_api_url,
        headers=dspa_auth_headers,
        pipeline_id=automl_managed_pipeline["pipeline_id"],
        pipeline_version_id=automl_managed_pipeline["pipeline_version_id"],
        run_name=f"automl-{task_type}-smoke-{pipelines_namespace.name}",
        parameters=parameters,
        ca_bundle=dspa_ca_bundle_file,
    )

    yield run_id
    if os.getenv("SKIP_TEARDOWN", "").lower() not in ("true", "1", "yes"):
        delete_pipeline_run(
            api_url=dspa_api_url,
            headers=dspa_auth_headers,
            run_id=run_id,
            ca_bundle=dspa_ca_bundle_file,
        )
