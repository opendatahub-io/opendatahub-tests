import pytest
from kubernetes.client.rest import ApiException
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.pipelines_components.constants import AUTOML_PIPELINE_START_TIMEOUT, PIPELINE_POLL_INTERVAL
from tests.pipelines_components.utils import (
    WORKFLOW_SUCCEEDED,
    collect_pipeline_pod_logs,
    get_pipeline_run,
    get_workflow_phase,
)


@pytest.mark.smoke
class TestAutoMLSmoke:
    """Verify managed AutoGluon tabular and timeseries pipelines start processing runs."""

    @pytest.mark.parametrize(
        "task_type",
        ["regression", "classification", "multiclass", "timeseries"],
        ids=["test_regression", "test_classification", "test_multiclass", "test_timeseries"],
    )
    def test_automl_pipeline_starts(
        self,
        task_type: str,
        admin_client: DynamicClient,
        pipelines_namespace: Namespace,
        dspa_api_url: str,
        dspa_auth_headers: dict[str, str],
        dspa_ca_bundle_file: str,
        automl_managed_pipeline: dict[str, str],
        automl_train_data: str,
        automl_run_id: str,
    ) -> None:
        """Given uploaded training data, when a managed run is submitted, then it exists and starts progressing."""
        run = get_pipeline_run(
            api_url=dspa_api_url,
            headers=dspa_auth_headers,
            run_id=automl_run_id,
            ca_bundle=dspa_ca_bundle_file,
        )
        assert run["run_id"] == automl_run_id, f"AutoML {task_type} run {automl_run_id} was not registered"
        pipeline_reference = run.get("pipeline_version_reference", {})
        assert pipeline_reference.get("pipeline_id") == automl_managed_pipeline["pipeline_id"], (
            f"AutoML {task_type} run does not reference the managed pipeline"
        )
        assert pipeline_reference.get("pipeline_version_id") == automl_managed_pipeline["pipeline_version_id"], (
            f"AutoML {task_type} run does not reference the managed pipeline version"
        )
        assert run.get("runtime_config", {}).get("parameters", {}).get("train_data_file_key") == automl_train_data, (
            f"AutoML {task_type} run does not reference the uploaded dataset"
        )

        last_phase: str | None = None
        try:
            for phase in TimeoutSampler(
                wait_timeout=AUTOML_PIPELINE_START_TIMEOUT,
                sleep=PIPELINE_POLL_INTERVAL,
                func=get_workflow_phase,
                exceptions_dict={ApiException: [], ConnectionError: [], TimeoutError: []},
                admin_client=admin_client,
                namespace=pipelines_namespace.name,
                run_id=automl_run_id,
            ):
                last_phase = phase
                if phase in ("Running", WORKFLOW_SUCCEEDED):
                    return
                if phase in ("Failed", "Error"):
                    collect_pipeline_pod_logs(
                        admin_client=admin_client,
                        namespace=pipelines_namespace.name,
                        run_id=automl_run_id,
                    )
                    pytest.fail(f"AutoML {task_type} run {automl_run_id} failed before progressing: {phase}")
        except TimeoutExpiredError:
            pytest.fail(
                f"AutoML {task_type} run {automl_run_id} did not start within "
                f"{AUTOML_PIPELINE_START_TIMEOUT}s (last workflow phase: {last_phase})"
            )
        pytest.fail(f"AutoML {task_type} run {automl_run_id} stopped being polled in phase {last_phase}")
