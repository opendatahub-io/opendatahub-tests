import os
import shlex
import uuid
from collections.abc import Generator
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.data_science_pipelines_application import DataSciencePipelinesApplication
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from timeout_sampler import TimeoutExpiredError

from tests.pipelines_components.constants import (
    AUTOML_PIPELINE_YAML,
    AUTOML_S3_BUCKET,
    AUTOML_TASK_CONFIGS,
    AUTOML_TIMESERIES_CONFIG,
    AUTOML_TIMESERIES_TRAIN_DATA_FILE_KEY,
    AUTOML_TRAIN_DATA_FILE_KEY,
    DSPA_NAME,
    DSPA_READY_BUFFER_SECONDS,
    DSPA_S3_BUCKET,
    DSPA_S3_SECRET,
    EXTERNAL_S3_SECRET,
    MANAGED_PIPELINE_AUTOML_TABULAR,
    MANAGED_PIPELINE_AUTOML_TIMESERIES,
    MANAGED_PIPELINE_POLL_INTERVAL,
    MANAGED_PIPELINE_WAIT_TIMEOUT,
    MINIO_MC_IMAGE,
    MINIO_UPLOADER_SECURITY_CONTEXT,
)
from tests.pipelines_components.utils import (
    create_pipeline_run,
    create_pipeline_run_managed,
    delete_pipeline,
    delete_pipeline_run,
    resolve_pipeline_yaml,
    upload_pipeline,
    use_managed_pipelines,
    wait_for_managed_pipeline,
)
from utilities.general import collect_pod_information

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="session", autouse=True)
def _validate_automl_env() -> None:
    if AUTOML_PIPELINE_YAML:
        LOGGER.info("AUTOML_PIPELINE_YAML is set — using legacy YAML upload mode")
    else:
        LOGGER.info("AUTOML_PIPELINE_YAML is not set — using managed pipeline mode")


@pytest.fixture(scope="class")
def automl_managed_pipeline(
    dspa: DataSciencePipelinesApplication,
    dspa_api_url: str,
    dspa_auth_headers: dict[str, str],
    dspa_ca_bundle_file: str,
) -> dict[str, str] | None:
    """Discovered managed pipeline info, or None in legacy mode."""
    if not use_managed_pipelines(yaml_env_value=AUTOML_PIPELINE_YAML):
        return None
    return wait_for_managed_pipeline(
        api_url=dspa_api_url,
        headers=dspa_auth_headers,
        display_name=MANAGED_PIPELINE_AUTOML_TABULAR,
        ca_bundle=dspa_ca_bundle_file,
        timeout=DSPA_READY_BUFFER_SECONDS + MANAGED_PIPELINE_WAIT_TIMEOUT,
        poll_interval=MANAGED_PIPELINE_POLL_INTERVAL,
    )


@pytest.fixture(scope="class")
def automl_pipeline_yaml_path() -> str | None:
    """Resolve the AutoML pipeline YAML to a local file path. None in managed mode."""
    if not AUTOML_PIPELINE_YAML:
        return None
    return resolve_pipeline_yaml(value=AUTOML_PIPELINE_YAML)


@pytest.fixture(scope="class")
def automl_pipeline_id(
    dspa_api_url: str,
    dspa_auth_headers: dict[str, str],
    dspa_ca_bundle_file: str,
    automl_pipeline_yaml_path: str | None,
    automl_managed_pipeline: dict[str, str] | None,
    pipelines_namespace: Namespace,
) -> Generator[str, Any, Any]:
    """Pipeline ID — from managed discovery or YAML upload."""
    if automl_managed_pipeline is not None:
        yield automl_managed_pipeline["pipeline_id"]
    else:
        assert automl_pipeline_yaml_path is not None, "AUTOML_PIPELINE_YAML must be set for legacy mode"
        pipeline_id = upload_pipeline(
            api_url=dspa_api_url,
            headers=dspa_auth_headers,
            pipeline_yaml_path=automl_pipeline_yaml_path,
            pipeline_name=f"automl-smoke-{pipelines_namespace.name}",
            ca_bundle=dspa_ca_bundle_file,
        )
        yield pipeline_id
        delete_pipeline(
            api_url=dspa_api_url,
            headers=dspa_auth_headers,
            pipeline_id=pipeline_id,
            ca_bundle=dspa_ca_bundle_file,
        )


@pytest.fixture(scope="function")
def automl_run_id(
    dspa_api_url: str,
    dspa_auth_headers: dict[str, str],
    dspa_ca_bundle_file: str,
    automl_pipeline_id: str,
    automl_managed_pipeline: dict[str, str] | None,
    pipelines_namespace: Namespace,
    task_type: str,
) -> Generator[str, Any, Any]:
    """Create a pipeline run and yield the run ID. Deletes the run on teardown."""
    task_config = AUTOML_TASK_CONFIGS[task_type]

    parameters: dict[str, Any] = {
        "train_data_secret_name": DSPA_S3_SECRET,
        "train_data_bucket_name": DSPA_S3_BUCKET,
        "train_data_file_key": AUTOML_TRAIN_DATA_FILE_KEY,
        "label_column": task_config["label_column"],
        "task_type": task_config["task_type"],
        "top_n": task_config["top_n"],
    }

    if automl_managed_pipeline is not None:
        run_id = create_pipeline_run_managed(
            api_url=dspa_api_url,
            headers=dspa_auth_headers,
            pipeline_id=automl_managed_pipeline["pipeline_id"],
            pipeline_version_id=automl_managed_pipeline["pipeline_version_id"],
            run_name=f"automl-smoke-{pipelines_namespace.name}",
            parameters=parameters,
            ca_bundle=dspa_ca_bundle_file,
        )
    else:
        run_id = create_pipeline_run(
            api_url=dspa_api_url,
            headers=dspa_auth_headers,
            pipeline_id=automl_pipeline_id,
            run_name=f"automl-smoke-{pipelines_namespace.name}",
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


# ---------------------------------------------------------------------------
# Timeseries fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def timeseries_managed_pipeline(
    dspa: DataSciencePipelinesApplication,
    dspa_api_url: str,
    dspa_auth_headers: dict[str, str],
    dspa_ca_bundle_file: str,
) -> dict[str, str]:
    """Discover the managed timeseries pipeline."""
    return wait_for_managed_pipeline(
        api_url=dspa_api_url,
        headers=dspa_auth_headers,
        display_name=MANAGED_PIPELINE_AUTOML_TIMESERIES,
        ca_bundle=dspa_ca_bundle_file,
        timeout=DSPA_READY_BUFFER_SECONDS + MANAGED_PIPELINE_WAIT_TIMEOUT,
        poll_interval=MANAGED_PIPELINE_POLL_INTERVAL,
    )


@pytest.fixture(scope="class")
def timeseries_pipeline_id(
    timeseries_managed_pipeline: dict[str, str],
) -> str:
    """Pipeline ID for the timeseries pipeline."""
    return timeseries_managed_pipeline["pipeline_id"]


@pytest.fixture(scope="function")
def timeseries_train_data(
    admin_client: DynamicClient,
    pipelines_namespace: Namespace,
    dspa_s3_credentials: Secret,
    external_s3_secret: Secret,
) -> str:
    """Download timeseries training CSV from external S3 and upload to DSPA MinIO."""
    env_var = "AUTOML_TIMESERIES_S3_TRAIN_DATA_KEY"
    src_key_value = os.environ.get(env_var)
    assert src_key_value, (
        f"Environment variable '{env_var}' is not set. "
        f"Set it in .env or shell to provide the S3 key for timeseries training data."
    )

    src_bucket = shlex.quote(s=AUTOML_S3_BUCKET)
    src_key = shlex.quote(s=src_key_value)
    dst_bucket = shlex.quote(s=DSPA_S3_BUCKET)
    dst_key = shlex.quote(s=AUTOML_TIMESERIES_TRAIN_DATA_FILE_KEY)

    minio_endpoint = f"http://minio-{DSPA_NAME}.{pipelines_namespace.name}.svc.cluster.local:9000"
    src_endpoint = os.environ.get("AWS_S3_ENDPOINT", "https://s3.amazonaws.com")

    script = (
        "export MC_CONFIG_DIR=/work/.mc && "
        "mc alias set src $SRC_ENDPOINT $AWS_ACCESS_KEY_ID $AWS_SECRET_ACCESS_KEY && "
        "mc alias set dspa $DST_ENDPOINT $DST_ACCESS_KEY $DST_SECRET_KEY && "
        f"mc cp src/{src_bucket}/{src_key} /work/train.csv && "
        f"mc cp /work/train.csv dspa/{dst_bucket}/{dst_key}"
    )

    pod_name = f"ts-data-uploader-{uuid.uuid4().hex[:8]}"
    with Pod(
        client=admin_client,
        name=pod_name,
        namespace=pipelines_namespace.name,
        restart_policy="Never",
        volumes=[{"name": "work", "emptyDir": {}}],
        containers=[
            {
                "name": "minio-uploader",
                "image": MINIO_MC_IMAGE,
                "command": ["/bin/sh", "-c"],
                "args": [script],
                "volumeMounts": [{"name": "work", "mountPath": "/work"}],
                "securityContext": MINIO_UPLOADER_SECURITY_CONTEXT,
                "env": [
                    {"name": "SRC_ENDPOINT", "value": src_endpoint},
                    {
                        "name": "AWS_ACCESS_KEY_ID",
                        "valueFrom": {"secretKeyRef": {"name": EXTERNAL_S3_SECRET, "key": "AWS_ACCESS_KEY_ID"}},
                    },
                    {
                        "name": "AWS_SECRET_ACCESS_KEY",
                        "valueFrom": {"secretKeyRef": {"name": EXTERNAL_S3_SECRET, "key": "AWS_SECRET_ACCESS_KEY"}},
                    },
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
    ) as upload_pod:
        try:
            upload_pod.wait_for_status(status="Succeeded", timeout=120)
        except TimeoutExpiredError:
            collect_pod_information(pod=upload_pod)
            raise

    return AUTOML_TIMESERIES_TRAIN_DATA_FILE_KEY


@pytest.fixture(scope="function")
def timeseries_run_id(
    dspa_api_url: str,
    dspa_auth_headers: dict[str, str],
    dspa_ca_bundle_file: str,
    timeseries_pipeline_id: str,
    timeseries_managed_pipeline: dict[str, str],
    pipelines_namespace: Namespace,
) -> Generator[str, Any, Any]:
    """Create a timeseries pipeline run and yield the run ID."""
    parameters: dict[str, Any] = {
        "train_data_secret_name": DSPA_S3_SECRET,
        "train_data_bucket_name": DSPA_S3_BUCKET,
        "train_data_file_key": AUTOML_TIMESERIES_TRAIN_DATA_FILE_KEY,
        "id_column": AUTOML_TIMESERIES_CONFIG["id_column"],
        "timestamp_column": AUTOML_TIMESERIES_CONFIG["timestamp_column"],
        "target": AUTOML_TIMESERIES_CONFIG["target"],
        "prediction_length": AUTOML_TIMESERIES_CONFIG["prediction_length"],
        "top_n": AUTOML_TIMESERIES_CONFIG["top_n"],
        "known_covariates_names": AUTOML_TIMESERIES_CONFIG["known_covariates_names"],
    }

    run_id = create_pipeline_run_managed(
        api_url=dspa_api_url,
        headers=dspa_auth_headers,
        pipeline_id=timeseries_managed_pipeline["pipeline_id"],
        pipeline_version_id=timeseries_managed_pipeline["pipeline_version_id"],
        run_name=f"automl-ts-smoke-{pipelines_namespace.name}",
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
