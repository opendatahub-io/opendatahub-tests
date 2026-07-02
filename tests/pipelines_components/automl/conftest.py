import os
from collections.abc import Generator
from typing import Any

import pytest
import structlog
from ocp_resources.data_science_pipelines_application import DataSciencePipelinesApplication
from ocp_resources.namespace import Namespace

from tests.pipelines_components.constants import (
    AUTOML_PIPELINE_YAML,
    AUTOML_TASK_CONFIGS,
    AUTOML_TRAIN_DATA_FILE_KEY,
    DSPA_READY_BUFFER_SECONDS,
    DSPA_S3_BUCKET,
    DSPA_S3_SECRET,
    MANAGED_PIPELINE_AUTOML_TABULAR,
    MANAGED_PIPELINE_POLL_INTERVAL,
    MANAGED_PIPELINE_WAIT_TIMEOUT,
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
