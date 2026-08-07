"""AutoML upgrade tests (regression + timeseries).

Pre-upgrade tests deploy a DSPA, run regression and timeseries pipelines,
deploy the trained models as InferenceServices, verify inference, and
capture baseline state to ConfigMaps.
Post-upgrade tests validate that the experiment runs, details, Argo
Workflows, managed pipelines, and deployed models all survived the
RHOAI upgrade.
"""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace

from tests.model_serving.model_runtime.autogluon.constant import ProtocolVersion  # noqa: NIT001
from tests.model_serving.model_runtime.autogluon.utils import (  # noqa: NIT001
    run_autogluon_inference,
    validate_deterministic_response,
)
from tests.pipelines_components.automl.upgrade.utils import REGRESSION_V2_INPUT, TIMESERIES_SUNSPOTS_V1_INPUT
from tests.pipelines_components.constants import AUTOML_PIPELINE_TIMEOUT
from tests.pipelines_components.utils import (
    WORKFLOW_SUCCEEDED,
    collect_pipeline_pod_logs,
    get_pipeline_run,
    get_workflow_completed_nodes,
    get_workflow_phase,
    wait_for_pipeline_run,
)
from utilities.constants import ModelVersion


@pytest.mark.usefixtures("pre_upgrade_pipelines_dsc_patch", "automl_capture_upgrade_baseline")
class TestPreUpgradeAutoML:
    """Run an AutoML regression experiment before upgrade, deploy the model, and capture baseline.

    Steps:
        0. Enable AI Pipelines in DSC (non-reverting)
        1. Create namespace and DSPA with MinIO
        2. Upload regression training data
        3. Create and run a regression pipeline
        4. Verify the pipeline completes successfully
        5. Verify the pipeline produced completed workflow nodes
        6. Deploy the trained model as an InferenceService
        7. Verify the model serves inference
        8. Save baseline to ConfigMap
    """

    @pytest.mark.dependency(name="automl_pre_upgrade_completes")
    @pytest.mark.pre_upgrade
    def test_automl_experiment_completes(
        self,
        admin_client: DynamicClient,
        upgrade_namespace: Namespace,
        upgrade_run_id: str,
    ) -> None:
        """Given a DSPA with training data, when a regression pipeline run is submitted, then it succeeds."""
        phase = wait_for_pipeline_run(
            admin_client=admin_client,
            namespace=upgrade_namespace.name,
            run_id=upgrade_run_id,
            timeout=AUTOML_PIPELINE_TIMEOUT,
        )

        if phase != WORKFLOW_SUCCEEDED:
            collect_pipeline_pod_logs(
                admin_client=admin_client,
                namespace=upgrade_namespace.name,
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
        upgrade_namespace: Namespace,
        upgrade_run_id: str,
    ) -> None:
        """Verify the completed pipeline has workflow nodes with execution records."""
        workflow_nodes = get_workflow_completed_nodes(
            admin_client=admin_client,
            namespace=upgrade_namespace.name,
            run_id=upgrade_run_id,
        )

        assert len(workflow_nodes) > 1, (
            f"Pipeline run {upgrade_run_id} has {len(workflow_nodes)} completed workflow nodes, "
            "expected multiple nodes for a multi-step AutoML pipeline"
        )

    @pytest.mark.dependency(name="automl_model_deployed", depends=["automl_pre_upgrade_completes"])
    @pytest.mark.pre_upgrade
    def test_automl_model_deployed(
        self,
        upgrade_inference_service: InferenceService,
    ) -> None:
        """Verify the trained model is deployed and the InferenceService is Ready."""
        assert upgrade_inference_service.exists, f"InferenceService {upgrade_inference_service.name} was not created"

    @pytest.mark.dependency(depends=["automl_model_deployed"])
    @pytest.mark.pre_upgrade
    def test_automl_model_scoring(
        self,
        upgrade_inference_service: InferenceService,
    ) -> None:
        """Send a V2 inference request to the deployed model and verify the response."""
        response = run_autogluon_inference(
            isvc=upgrade_inference_service,
            input_data=REGRESSION_V2_INPUT,
            protocol_version=ProtocolVersion.V2,
            model_version=ModelVersion.AUTOGLUON_1,
        )
        validate_deterministic_response(response=response)


class TestPostUpgradeAutoML:
    """Validate that the pre-upgrade AutoML experiment and model survived the RHOAI upgrade.

    Steps:
        1. Load baseline from ConfigMap
        2. Verify the pipeline run is accessible via KFP API
        3. Verify the run details are intact
        4. Verify the Argo Workflow CRD still exists
        5. Verify the workflow nodes survived
        6. Verify the managed pipeline is still discoverable
        7. Verify the InferenceService survived and is Ready
        8. Verify the model still serves inference
    """

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="automl_run_accessible")
    def test_automl_experiment_accessible(
        self,
        upgrade_dspa_api_url: str,
        upgrade_dspa_auth_headers: dict[str, str],
        upgrade_dspa_ca_bundle_file: str,
        automl_upgrade_baseline: dict,
    ) -> None:
        """Verify the pre-upgrade experiment run is accessible and still in SUCCEEDED state."""
        run_id = automl_upgrade_baseline["run_id"]

        run = get_pipeline_run(
            api_url=upgrade_dspa_api_url,
            headers=upgrade_dspa_auth_headers,
            run_id=run_id,
            ca_bundle=upgrade_dspa_ca_bundle_file,
        )

        run_state = run.get("state", "")
        assert "SUCCEEDED" in run_state.upper(), (
            f"Pipeline run {run_id} state is '{run_state}' after upgrade, expected SUCCEEDED"
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["automl_run_accessible"])
    def test_automl_experiment_details_intact(
        self,
        upgrade_dspa_api_url: str,
        upgrade_dspa_auth_headers: dict[str, str],
        upgrade_dspa_ca_bundle_file: str,
        automl_upgrade_baseline: dict,
    ) -> None:
        """Verify the run details (display name, pipeline reference, parameters) are intact."""
        run_id = automl_upgrade_baseline["run_id"]
        expected_display_name = automl_upgrade_baseline["run_display_name"]

        run = get_pipeline_run(
            api_url=upgrade_dspa_api_url,
            headers=upgrade_dspa_auth_headers,
            run_id=run_id,
            ca_bundle=upgrade_dspa_ca_bundle_file,
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
        upgrade_namespace: Namespace,
        automl_upgrade_baseline: dict,
    ) -> None:
        """Verify the Argo Workflow CRD still exists with Succeeded phase."""
        run_id = automl_upgrade_baseline["run_id"]

        phase = get_workflow_phase(
            admin_client=admin_client,
            namespace=upgrade_namespace.name,
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
        upgrade_namespace: Namespace,
        automl_upgrade_baseline: dict,
    ) -> None:
        """Verify the workflow execution nodes survived the upgrade."""
        run_id = automl_upgrade_baseline["run_id"]

        workflow_nodes = get_workflow_completed_nodes(
            admin_client=admin_client,
            namespace=upgrade_namespace.name,
            run_id=run_id,
        )

        assert len(workflow_nodes) > 1, (
            f"Pipeline run {run_id} has {len(workflow_nodes)} completed workflow nodes after upgrade, "
            "expected multiple nodes — execution records were lost"
        )

    @pytest.mark.post_upgrade
    def test_automl_managed_pipeline_accessible(
        self,
        upgrade_tabular_managed_pipeline: dict[str, str] | None,
    ) -> None:
        """Verify the managed AutoML pipeline is still discoverable after upgrade."""
        assert upgrade_tabular_managed_pipeline is not None, "Managed AutoML tabular pipeline not found after upgrade"
        assert upgrade_tabular_managed_pipeline.get("pipeline_id"), "Managed pipeline has no pipeline_id after upgrade"
        assert upgrade_tabular_managed_pipeline.get("pipeline_version_id"), (
            "Managed pipeline has no pipeline_version_id after upgrade"
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="automl_model_survived")
    def test_automl_model_survived_upgrade(
        self,
        upgrade_inference_service: InferenceService,
    ) -> None:
        """Verify the InferenceService still exists and is Ready after upgrade."""
        assert upgrade_inference_service.exists, (
            f"InferenceService {upgrade_inference_service.name} does not exist after upgrade"
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["automl_model_survived"])
    def test_automl_model_scoring_after_upgrade(
        self,
        upgrade_inference_service: InferenceService,
    ) -> None:
        """Verify the model still serves inference after upgrade."""
        response = run_autogluon_inference(
            isvc=upgrade_inference_service,
            input_data=REGRESSION_V2_INPUT,
            protocol_version=ProtocolVersion.V2,
            model_version=ModelVersion.AUTOGLUON_1,
        )
        validate_deterministic_response(response=response)


@pytest.mark.usefixtures("pre_upgrade_pipelines_dsc_patch", "ts_capture_upgrade_baseline")
class TestPreUpgradeAutoMLTimeseries:
    """Run an AutoML timeseries experiment before upgrade, deploy the model, and capture baseline."""

    @pytest.mark.dependency(name="ts_pre_upgrade_completes")
    @pytest.mark.pre_upgrade
    def test_ts_experiment_completes(
        self,
        admin_client: DynamicClient,
        upgrade_namespace: Namespace,
        upgrade_ts_run_id: str,
    ) -> None:
        """Given a DSPA with timeseries data, when a timeseries pipeline run is submitted, then it succeeds."""
        phase = wait_for_pipeline_run(
            admin_client=admin_client,
            namespace=upgrade_namespace.name,
            run_id=upgrade_ts_run_id,
            timeout=AUTOML_PIPELINE_TIMEOUT,
        )

        if phase != WORKFLOW_SUCCEEDED:
            collect_pipeline_pod_logs(
                admin_client=admin_client,
                namespace=upgrade_namespace.name,
                run_id=upgrade_ts_run_id,
            )

        assert phase == WORKFLOW_SUCCEEDED, (
            f"AutoML timeseries pipeline run {upgrade_ts_run_id} ended with phase '{phase}', "
            f"expected '{WORKFLOW_SUCCEEDED}'"
        )

    @pytest.mark.dependency(depends=["ts_pre_upgrade_completes"])
    @pytest.mark.pre_upgrade
    def test_ts_experiment_has_artifacts(
        self,
        admin_client: DynamicClient,
        upgrade_namespace: Namespace,
        upgrade_ts_run_id: str,
    ) -> None:
        """Verify the completed timeseries pipeline has workflow nodes with execution records."""
        workflow_nodes = get_workflow_completed_nodes(
            admin_client=admin_client,
            namespace=upgrade_namespace.name,
            run_id=upgrade_ts_run_id,
        )

        assert len(workflow_nodes) > 1, (
            f"Timeseries pipeline run {upgrade_ts_run_id} has {len(workflow_nodes)} completed workflow nodes, "
            "expected multiple nodes"
        )

    @pytest.mark.dependency(name="ts_model_deployed", depends=["ts_pre_upgrade_completes"])
    @pytest.mark.pre_upgrade
    def test_ts_model_deployed(
        self,
        upgrade_ts_inference_service: InferenceService,
    ) -> None:
        """Verify the trained timeseries model is deployed and the InferenceService is Ready."""
        assert upgrade_ts_inference_service.exists, (
            f"InferenceService {upgrade_ts_inference_service.name} was not created"
        )

    @pytest.mark.dependency(depends=["ts_model_deployed"])
    @pytest.mark.pre_upgrade
    def test_ts_model_scoring(
        self,
        upgrade_ts_inference_service: InferenceService,
    ) -> None:
        """Send a V1 inference request to the deployed timeseries model and verify the response."""
        response = run_autogluon_inference(
            isvc=upgrade_ts_inference_service,
            input_data=TIMESERIES_SUNSPOTS_V1_INPUT,
            protocol_version=ProtocolVersion.V1,
            model_version=ModelVersion.AUTOGLUON_1,
        )
        validate_deterministic_response(response=response)


@pytest.mark.usefixtures("post_upgrade_pipelines_dsc_restore")
class TestPostUpgradeAutoMLTimeseries:
    """Validate that the pre-upgrade AutoML timeseries experiment and model survived the upgrade."""

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="ts_run_accessible")
    def test_ts_experiment_accessible(
        self,
        upgrade_dspa_api_url: str,
        upgrade_dspa_auth_headers: dict[str, str],
        upgrade_dspa_ca_bundle_file: str,
        ts_upgrade_baseline: dict,
    ) -> None:
        """Verify the pre-upgrade timeseries run is accessible and still in SUCCEEDED state."""
        run_id = ts_upgrade_baseline["run_id"]

        run = get_pipeline_run(
            api_url=upgrade_dspa_api_url,
            headers=upgrade_dspa_auth_headers,
            run_id=run_id,
            ca_bundle=upgrade_dspa_ca_bundle_file,
        )

        run_state = run.get("state", "")
        assert "SUCCEEDED" in run_state.upper(), (
            f"Timeseries pipeline run {run_id} state is '{run_state}' after upgrade, expected SUCCEEDED"
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["ts_run_accessible"])
    def test_ts_experiment_details_intact(
        self,
        upgrade_dspa_api_url: str,
        upgrade_dspa_auth_headers: dict[str, str],
        upgrade_dspa_ca_bundle_file: str,
        ts_upgrade_baseline: dict,
    ) -> None:
        """Verify the timeseries run details are intact after upgrade."""
        run_id = ts_upgrade_baseline["run_id"]
        expected_display_name = ts_upgrade_baseline["run_display_name"]

        run = get_pipeline_run(
            api_url=upgrade_dspa_api_url,
            headers=upgrade_dspa_auth_headers,
            run_id=run_id,
            ca_bundle=upgrade_dspa_ca_bundle_file,
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
    @pytest.mark.dependency(depends=["ts_run_accessible"])
    def test_ts_workflow_survived(
        self,
        admin_client: DynamicClient,
        upgrade_namespace: Namespace,
        ts_upgrade_baseline: dict,
    ) -> None:
        """Verify the timeseries Argo Workflow CRD still exists with Succeeded phase."""
        run_id = ts_upgrade_baseline["run_id"]

        phase = get_workflow_phase(
            admin_client=admin_client,
            namespace=upgrade_namespace.name,
            run_id=run_id,
        )

        assert phase == WORKFLOW_SUCCEEDED, (
            f"Argo Workflow for timeseries run {run_id} has phase '{phase}' after upgrade, "
            f"expected '{WORKFLOW_SUCCEEDED}'"
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["ts_run_accessible"])
    def test_ts_artifacts_survived(
        self,
        admin_client: DynamicClient,
        upgrade_namespace: Namespace,
        ts_upgrade_baseline: dict,
    ) -> None:
        """Verify the timeseries workflow execution nodes survived the upgrade."""
        run_id = ts_upgrade_baseline["run_id"]

        workflow_nodes = get_workflow_completed_nodes(
            admin_client=admin_client,
            namespace=upgrade_namespace.name,
            run_id=run_id,
        )

        assert len(workflow_nodes) > 1, (
            f"Timeseries pipeline run {run_id} has {len(workflow_nodes)} completed workflow nodes after upgrade"
        )

    @pytest.mark.post_upgrade
    def test_ts_managed_pipeline_accessible(
        self,
        ts_managed_pipeline: dict[str, str],
    ) -> None:
        """Verify the managed timeseries pipeline is still discoverable after upgrade."""
        assert ts_managed_pipeline.get("pipeline_id"), "Managed timeseries pipeline has no pipeline_id after upgrade"
        assert ts_managed_pipeline.get("pipeline_version_id"), (
            "Managed timeseries pipeline has no pipeline_version_id after upgrade"
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="ts_model_survived")
    def test_ts_model_survived_upgrade(
        self,
        upgrade_ts_inference_service: InferenceService,
    ) -> None:
        """Verify the timeseries InferenceService still exists after upgrade."""
        assert upgrade_ts_inference_service.exists, (
            f"InferenceService {upgrade_ts_inference_service.name} does not exist after upgrade"
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["ts_model_survived"])
    def test_ts_model_scoring_after_upgrade(
        self,
        upgrade_ts_inference_service: InferenceService,
    ) -> None:
        """Verify the timeseries model still serves inference after upgrade."""
        response = run_autogluon_inference(
            isvc=upgrade_ts_inference_service,
            input_data=TIMESERIES_SUNSPOTS_V1_INPUT,
            protocol_version=ProtocolVersion.V1,
            model_version=ModelVersion.AUTOGLUON_1,
        )
        validate_deterministic_response(response=response)
