import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret

from tests.pipelines_components.constants import AUTOML_PIPELINE_TIMEOUT
from tests.pipelines_components.utils import (
    WORKFLOW_SUCCEEDED,
    collect_pipeline_pod_logs,
    wait_for_pipeline_run,
)


@pytest.mark.smoke
class TestAutoMLSmoke:
    """AutoML pipeline smoke tests using AutoGluon Tabular Training from pipelines-components."""

    @pytest.mark.parametrize(
        "task_type", ["regression", "classification", "multiclass"], ids=["regression", "classification", "multiclass"]
    )
    def test_automl_pipeline_completes(
        self,
        task_type: str,
        admin_client: DynamicClient,
        pipelines_namespace: Namespace,
        dspa_s3_credentials: Secret,
        automl_train_data: str,
        automl_run_id: str,
    ) -> None:
        """Given a DSPA with training data in S3, when an AutoML pipeline run is submitted, then it succeeds."""
        phase = wait_for_pipeline_run(
            admin_client=admin_client,
            namespace=pipelines_namespace.name,
            run_id=automl_run_id,
            timeout=AUTOML_PIPELINE_TIMEOUT,
        )

        if phase != WORKFLOW_SUCCEEDED:
            collect_pipeline_pod_logs(
                admin_client=admin_client,
                namespace=pipelines_namespace.name,
                run_id=automl_run_id,
            )

        assert phase == WORKFLOW_SUCCEEDED, (
            f"AutoML {task_type} pipeline run {automl_run_id} ended with phase '{phase}', "
            f"expected '{WORKFLOW_SUCCEEDED}'"
        )

    def test_timeseries_pipeline_completes(
        self,
        admin_client: DynamicClient,
        pipelines_namespace: Namespace,
        dspa_s3_credentials: Secret,
        timeseries_train_data: str,
        timeseries_run_id: str,
    ) -> None:
        """Given a DSPA with timeseries data in S3, when a timeseries pipeline run is submitted, then it succeeds."""
        phase = wait_for_pipeline_run(
            admin_client=admin_client,
            namespace=pipelines_namespace.name,
            run_id=timeseries_run_id,
            timeout=AUTOML_PIPELINE_TIMEOUT,
        )

        if phase != WORKFLOW_SUCCEEDED:
            collect_pipeline_pod_logs(
                admin_client=admin_client,
                namespace=pipelines_namespace.name,
                run_id=timeseries_run_id,
            )

        assert phase == WORKFLOW_SUCCEEDED, (
            f"AutoML timeseries pipeline run {timeseries_run_id} ended with phase '{phase}', "
            f"expected '{WORKFLOW_SUCCEEDED}'"
        )
