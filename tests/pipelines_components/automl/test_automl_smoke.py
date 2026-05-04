import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret

from tests.pipelines_components.constants import (
    AUTOML_LABEL_COLUMN,
    AUTOML_PIPELINE_TIMEOUT,
    AUTOML_PIPELINE_YAML,
    AUTOML_TASK_TYPE,
    AUTOML_TOP_N,
    AUTOML_TRAIN_DATA_FILE_KEY,
    DSPA_S3_BUCKET,
    DSPA_S3_SECRET,
)
from tests.pipelines_components.utils import (
    WORKFLOW_SUCCEEDED,
    collect_pipeline_pod_logs,
    create_pipeline_run,
    delete_pipeline,
    delete_pipeline_run,
    upload_pipeline,
    wait_for_pipeline_run,
)


@pytest.mark.smoke
class TestAutoMLSmoke:
    """AutoML pipeline smoke tests using AutoGluon Tabular Training from pipelines-components."""

    def test_automl_pipeline_completes(
        self,
        admin_client: DynamicClient,
        dspa_api_url: str,
        dspa_auth_headers: dict[str, str],
        dspa_ca_bundle_file: str,
        pipelines_namespace: Namespace,
        dspa_s3_credentials: Secret,
        automl_train_data: str,
    ) -> None:
        """Given a DSPA with training data in S3, when an AutoML pipeline run is submitted, then it succeeds."""
        pipeline_id = None
        run_id = None

        try:
            pipeline_id = upload_pipeline(
                api_url=dspa_api_url,
                headers=dspa_auth_headers,
                pipeline_yaml_path=AUTOML_PIPELINE_YAML,
                pipeline_name=f"automl-smoke-{pipelines_namespace.name}",
                ca_bundle=dspa_ca_bundle_file,
            )

            run_id = create_pipeline_run(
                api_url=dspa_api_url,
                headers=dspa_auth_headers,
                pipeline_id=pipeline_id,
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

            try:
                phase = wait_for_pipeline_run(
                    admin_client=admin_client,
                    namespace=pipelines_namespace.name,
                    run_id=run_id,
                    timeout=AUTOML_PIPELINE_TIMEOUT,
                )
            except TimeoutError:
                collect_pipeline_pod_logs(
                    admin_client=admin_client,
                    namespace=pipelines_namespace.name,
                    run_id=run_id,
                )
                raise

            if phase != WORKFLOW_SUCCEEDED:
                collect_pipeline_pod_logs(
                    admin_client=admin_client,
                    namespace=pipelines_namespace.name,
                    run_id=run_id,
                )

            assert phase == WORKFLOW_SUCCEEDED, (
                f"AutoML pipeline run {run_id} ended with phase '{phase}', expected '{WORKFLOW_SUCCEEDED}'"
            )
        finally:
            if run_id:
                delete_pipeline_run(
                    api_url=dspa_api_url,
                    headers=dspa_auth_headers,
                    run_id=run_id,
                    ca_bundle=dspa_ca_bundle_file,
                )
            if pipeline_id:
                delete_pipeline(
                    api_url=dspa_api_url,
                    headers=dspa_auth_headers,
                    pipeline_id=pipeline_id,
                    ca_bundle=dspa_ca_bundle_file,
                )
