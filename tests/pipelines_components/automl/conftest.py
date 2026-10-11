import os
from collections.abc import Generator
from typing import Any

import pytest
from ocp_resources.data_science_pipelines_application import DataSciencePipelinesApplication
from ocp_resources.namespace import Namespace

from tests.pipelines_components.constants import (
    AUTOML_TASK_CONFIGS,
    AUTOML_TIMESERIES_CONFIG,
    DSPA_READY_BUFFER_SECONDS,
    DSPA_S3_BUCKET,
    DSPA_S3_SECRET,
    MANAGED_PIPELINE_AUTOML_TABULAR,
    MANAGED_PIPELINE_AUTOML_TIMESERIES,
    MANAGED_PIPELINE_POLL_INTERVAL,
    MANAGED_PIPELINE_WAIT_TIMEOUT,
)
from tests.pipelines_components.utils import create_pipeline_run_managed, delete_pipeline_run, wait_for_managed_pipeline


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
