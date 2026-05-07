from collections.abc import Generator
from typing import Any

import pytest
import structlog

from tests.pipelines_components.constants import (
    AUTOML_LABEL_COLUMN,
    AUTOML_PIPELINE_YAML,
    AUTOML_TASK_TYPE,
    AUTOML_TOP_N,
    AUTOML_TRAIN_DATA_FILE_KEY,
    DSPA_S3_BUCKET,
    DSPA_S3_SECRET,
)
from tests.pipelines_components.utils import (
    create_pipeline_run,
    delete_pipeline,
    delete_pipeline_run,
    resolve_pipeline_yaml,
    upload_pipeline,
)

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="session", autouse=True)
def _validate_automl_env() -> None:
    if not AUTOML_PIPELINE_YAML:
        pytest.skip(
            "AutoML smoke test requires AUTOML_PIPELINE_YAML environment variable.\n"
            "Set it to a local path or URL of a compiled AutoGluon Tabular Training pipeline YAML."
        )


@pytest.fixture(scope="class")
def automl_pipeline_yaml_path() -> str:
    """Resolve the AutoML pipeline YAML to a local file path (downloads if URL)."""
    return resolve_pipeline_yaml(value=AUTOML_PIPELINE_YAML)


@pytest.fixture(scope="class")
def automl_pipeline_id(
    dspa_api_url: str,
    dspa_auth_headers: dict[str, str],
    dspa_ca_bundle_file: str,
    automl_pipeline_yaml_path: str,
    pipelines_namespace: Namespace,  # noqa: F821
) -> Generator[str, Any, Any]:
    """Upload the AutoML pipeline YAML and yield the pipeline ID. Deletes the pipeline on teardown."""
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


@pytest.fixture(scope="class")
def automl_run_id(
    dspa_api_url: str,
    dspa_auth_headers: dict[str, str],
    dspa_ca_bundle_file: str,
    automl_pipeline_id: str,
    pipelines_namespace: Namespace,  # noqa: F821
) -> Generator[str, Any, Any]:
    """Create a pipeline run and yield the run ID. Deletes the run on teardown."""
    run_id = create_pipeline_run(
        api_url=dspa_api_url,
        headers=dspa_auth_headers,
        pipeline_id=automl_pipeline_id,
        run_name=f"automl-smoke-{pipelines_namespace.name}",
        parameters={
            "train_data_secret_name": DSPA_S3_SECRET,
            "train_data_bucket_name": DSPA_S3_BUCKET,
            "train_data_file_key": AUTOML_TRAIN_DATA_FILE_KEY,
            "label_column": AUTOML_LABEL_COLUMN,
            "task_type": AUTOML_TASK_TYPE,
            "top_n": AUTOML_TOP_N,
        },
        ca_bundle=dspa_ca_bundle_file,
    )
    yield run_id
    delete_pipeline_run(
        api_url=dspa_api_url,
        headers=dspa_auth_headers,
        run_id=run_id,
        ca_bundle=dspa_ca_bundle_file,
    )
