"""AutoML upgrade tests.

Pre-upgrade tests deploy a DSPA, run a regression pipeline, verify it
produces artifacts, and capture baseline state to a ConfigMap.
Post-upgrade tests validate that the experiment run, its details, the
Argo Workflow, artifacts, and the managed pipeline survived the RHOAI upgrade.

TODO(RHOAIENG-70979): Add model deployment and scoring tests.
    The acceptance criteria require deploying the AutoML-trained model as an
    InferenceService and running inference against it — both pre-upgrade
    (to verify it works) and post-upgrade (to verify it survived).
    This requires:
      1. Extracting the model artifact S3 URI from the workflow outputs
      2. Determining the correct serving runtime for AutoGluon models
      3. Creating an InferenceService via utilities.inference_utils.create_isvc()
      4. Sending inference requests and validating responses
    There is no existing precedent in the repo for deploying models from
    pipeline outputs, so this is deferred to a follow-up.
"""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace

from tests.pipelines_components.constants import AUTOML_PIPELINE_TIMEOUT
from tests.pipelines_components.utils import (
    WORKFLOW_SUCCEEDED,
    collect_pipeline_pod_logs,
    get_pipeline_run,
    get_workflow_completed_nodes,
    get_workflow_phase,
    wait_for_pipeline_run,
)


@pytest.mark.usefixtures("pre_upgrade_pipelines_dsc_patch", "automl_capture_upgrade_baseline")
class TestPreUpgradeAutoML:
    """Run an AutoML regression experiment before upgrade and capture baseline.

    Steps:
        0. Enable AI Pipelines in DSC (non-reverting)
        1. Create namespace and DSPA with MinIO
        2. Upload regression training data
        3. Create and run a regression pipeline
        4. Verify the pipeline completes successfully
        5. Verify the pipeline produced output artifacts
        6. Save run_id and pipeline details to ConfigMap
    """

    @pytest.mark.dependency(name="automl_pre_upgrade_completes")
    @pytest.mark.pre_upgrade
    def test_automl_experiment_completes(
        self,
        admin_client: DynamicClient,
        pipelines_namespace: Namespace,
        upgrade_run_id: str,
    ) -> None:
        """Given a DSPA with training data, when a regression pipeline run is submitted, then it succeeds."""
        phase = wait_for_pipeline_run(
            admin_client=admin_client,
            namespace=pipelines_namespace.name,
            run_id=upgrade_run_id,
            timeout=AUTOML_PIPELINE_TIMEOUT,
        )

        if phase != WORKFLOW_SUCCEEDED:
            collect_pipeline_pod_logs(
                admin_client=admin_client,
                namespace=pipelines_namespace.name,
                run_id=upgrade_run_id,
            )

        assert phase == WORKFLOW_SUCCEEDED, (
            f"AutoML upgrade regression pipeline run {upgrade_run_id} ended with phase '{phase}', "
            f"expected '{WORKFLOW_SUCCEEDED}'"
        )

    @pytest.mark.dependency(depends=["automl_pre_upgrade_completes"])
    @pytest.mark.pre_upgrade
    def test_automl_experiment_has_artifacts(
        self,
        admin_client: DynamicClient,
        pipelines_namespace: Namespace,
        upgrade_run_id: str,
    ) -> None:
        """Verify the completed pipeline has workflow nodes with execution records."""
        workflow_nodes = get_workflow_completed_nodes(
            admin_client=admin_client,
            namespace=pipelines_namespace.name,
            run_id=upgrade_run_id,
        )

        assert len(workflow_nodes) > 1, (
            f"Pipeline run {upgrade_run_id} has {len(workflow_nodes)} completed workflow nodes, "
            "expected multiple nodes for a multi-step AutoML pipeline"
        )


class TestPostUpgradeAutoML:
    """Validate that the pre-upgrade AutoML experiment survived the RHOAI upgrade.

    Steps:
        1. Load baseline from ConfigMap
        2. Verify the pipeline run is accessible via KFP API
        3. Verify the run details are intact
        4. Verify the Argo Workflow CRD still exists
        5. Verify the workflow artifacts survived
        6. Verify the managed pipeline is still discoverable
    """

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="automl_run_accessible")
    def test_automl_experiment_accessible(
        self,
        dspa_api_url: str,
        dspa_auth_headers: dict[str, str],
        dspa_ca_bundle_file: str,
        automl_upgrade_baseline: dict,
    ) -> None:
        """Verify the pre-upgrade experiment run is accessible and still in SUCCEEDED state."""
        run_id = automl_upgrade_baseline["run_id"]

        run = get_pipeline_run(
            api_url=dspa_api_url,
            headers=dspa_auth_headers,
            run_id=run_id,
            ca_bundle=dspa_ca_bundle_file,
        )

        run_state = run.get("state", "")
        assert "SUCCEEDED" in run_state.upper(), (
            f"Pipeline run {run_id} state is '{run_state}' after upgrade, expected SUCCEEDED"
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["automl_run_accessible"])
    def test_automl_experiment_details_intact(
        self,
        dspa_api_url: str,
        dspa_auth_headers: dict[str, str],
        dspa_ca_bundle_file: str,
        automl_upgrade_baseline: dict,
    ) -> None:
        """Verify the run details (display name, pipeline reference, parameters) are intact."""
        run_id = automl_upgrade_baseline["run_id"]
        expected_display_name = automl_upgrade_baseline["run_display_name"]

        run = get_pipeline_run(
            api_url=dspa_api_url,
            headers=dspa_auth_headers,
            run_id=run_id,
            ca_bundle=dspa_ca_bundle_file,
        )

        assert run.get("display_name") == expected_display_name, (
            f"Run display_name changed: expected '{expected_display_name}', got '{run.get('display_name')}'"
        )

        assert run.get("pipeline_version_reference"), (
            f"Pipeline version reference missing from run {run_id} after upgrade"
        )

        runtime_config = run.get("runtime_config", {})
        assert runtime_config.get("parameters"), f"Runtime config parameters missing from run {run_id} after upgrade"

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["automl_run_accessible"])
    def test_automl_workflow_survived(
        self,
        admin_client: DynamicClient,
        pipelines_namespace: Namespace,
        automl_upgrade_baseline: dict,
    ) -> None:
        """Verify the Argo Workflow CRD still exists with Succeeded phase."""
        run_id = automl_upgrade_baseline["run_id"]

        phase = get_workflow_phase(
            admin_client=admin_client,
            namespace=pipelines_namespace.name,
            run_id=run_id,
        )

        assert phase == WORKFLOW_SUCCEEDED, (
            f"Argo Workflow for run {run_id} has phase '{phase}' after upgrade, expected '{WORKFLOW_SUCCEEDED}'"
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["automl_run_accessible"])
    def test_automl_artifacts_survived(
        self,
        admin_client: DynamicClient,
        pipelines_namespace: Namespace,
        automl_upgrade_baseline: dict,
    ) -> None:
        """Verify the workflow execution nodes survived the upgrade."""
        run_id = automl_upgrade_baseline["run_id"]

        workflow_nodes = get_workflow_completed_nodes(
            admin_client=admin_client,
            namespace=pipelines_namespace.name,
            run_id=run_id,
        )

        assert len(workflow_nodes) > 1, (
            f"Pipeline run {run_id} has {len(workflow_nodes)} completed workflow nodes after upgrade, "
            "expected multiple nodes — execution records were lost"
        )

    @pytest.mark.post_upgrade
    def test_automl_managed_pipeline_accessible(
        self,
        automl_managed_pipeline: dict[str, str] | None,
    ) -> None:
        """Verify the managed AutoML pipeline is still discoverable after upgrade."""
        assert automl_managed_pipeline is not None, "Managed AutoML tabular pipeline not found after upgrade"
        assert automl_managed_pipeline.get("pipeline_id"), "Managed pipeline has no pipeline_id after upgrade"
        assert automl_managed_pipeline.get("pipeline_version_id"), (
            "Managed pipeline has no pipeline_version_id after upgrade"
        )
