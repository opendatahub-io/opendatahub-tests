"""AutoRAG upgrade tests.

Pre-upgrade tests deploy DSPA + OGX infrastructure, run the AutoRAG
optimization pipeline, verify it completes, and validate that the
resulting RAG pattern can query OGX. Baseline state is captured to a
ConfigMap.

Post-upgrade tests validate that the experiment runs, details, Argo
Workflows, managed pipeline, and RAG query all survived the RHOAI
upgrade.
"""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ogx_client import OgxClient

from tests.pipelines_components.autorag.upgrade.utils import assert_rag_query_works
from tests.pipelines_components.constants import AUTORAG_PIPELINE_TIMEOUT
from tests.pipelines_components.utils import (
    WORKFLOW_SUCCEEDED,
    collect_pipeline_pod_logs,
    get_pipeline_run,
    get_workflow_completed_nodes,
    get_workflow_phase,
    wait_for_pipeline_run,
)


@pytest.mark.usefixtures("pre_upgrade_autorag_dsc_patch", "autorag_capture_upgrade_baseline")
class TestPreUpgradeAutoRAG:
    """Run an AutoRAG experiment before upgrade and capture baseline.

    Steps:
        0. Enable AI Pipelines + OGX in DSC (non-reverting)
        1. Create namespace, DSPA, OGX stack, deploy vLLM models
        2. Upload test data to MinIO
        3. Create and run the AutoRAG optimization pipeline
        4. Verify the pipeline completes successfully
        5. Verify the pipeline produced completed workflow nodes
        6. Verify RAG pattern can query OGX via file_search
        7. Save baseline (run_id, vector_store_ids, pipeline IDs) to ConfigMap
    """

    @pytest.mark.dependency(name="autorag_pre_upgrade_completes")
    @pytest.mark.pre_upgrade
    def test_autorag_experiment_completes(
        self,
        admin_client: DynamicClient,
        upgrade_namespace: Namespace,
        upgrade_run_id: str,
    ) -> None:
        """Given a DSPA with documents and OGX, when an AutoRAG pipeline run is submitted, then it succeeds."""
        phase = wait_for_pipeline_run(
            admin_client=admin_client,
            namespace=upgrade_namespace.name,
            run_id=upgrade_run_id,
            timeout=AUTORAG_PIPELINE_TIMEOUT,
        )

        if phase != WORKFLOW_SUCCEEDED:
            collect_pipeline_pod_logs(
                admin_client=admin_client,
                namespace=upgrade_namespace.name,
                run_id=upgrade_run_id,
            )

        assert phase == WORKFLOW_SUCCEEDED, (
            f"AutoRAG upgrade pipeline run {upgrade_run_id} ended with phase '{phase}', "
            f"expected '{WORKFLOW_SUCCEEDED}'"
        )

    @pytest.mark.dependency(depends=["autorag_pre_upgrade_completes"])
    @pytest.mark.pre_upgrade
    def test_autorag_experiment_has_artifacts(
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
            "expected multiple nodes for a multi-step AutoRAG pipeline"
        )

    @pytest.mark.dependency(depends=["autorag_pre_upgrade_completes"])
    @pytest.mark.pre_upgrade
    def test_autorag_pattern_query(
        self,
        upgrade_ogx_client: OgxClient,
        upgrade_discovered_models: tuple[str, str],
    ) -> None:
        """Verify the AutoRAG pattern can query OGX via file_search after the pipeline completes."""
        _, generation_model = upgrade_discovered_models

        vector_stores = upgrade_ogx_client.vector_stores.list()
        assert vector_stores.data, "No vector stores found in OGX after pipeline completion"

        vector_store = vector_stores.data[0]
        assert_rag_query_works(
            ogx_client=upgrade_ogx_client,
            model_id=generation_model,
            vector_store=vector_store,
        )


@pytest.mark.usefixtures("post_upgrade_autorag_dsc_restore")
class TestPostUpgradeAutoRAG:
    """Validate that the pre-upgrade AutoRAG experiment and RAG pattern survived the RHOAI upgrade.

    Steps:
        1. Load baseline from ConfigMap
        2. Verify the pipeline run is accessible via KFP API
        3. Verify the run details are intact
        4. Verify the Argo Workflow CRD still exists
        5. Verify the workflow nodes survived
        6. Verify the managed pipeline is still discoverable
        7. Verify the RAG pattern can still query OGX via file_search
    """

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="autorag_run_accessible")
    def test_autorag_experiment_accessible(
        self,
        upgrade_dspa_api_url: str,
        upgrade_dspa_auth_headers: dict[str, str],
        upgrade_dspa_ca_bundle_file: str,
        autorag_upgrade_baseline: dict,
    ) -> None:
        """Verify the pre-upgrade experiment run is accessible and still in SUCCEEDED state."""
        run_id = autorag_upgrade_baseline["run_id"]

        run = get_pipeline_run(
            api_url=upgrade_dspa_api_url,
            headers=upgrade_dspa_auth_headers,
            run_id=run_id,
            ca_bundle=upgrade_dspa_ca_bundle_file,
        )

        run_state = run.get("state", "")
        assert "SUCCEEDED" in run_state.upper(), (
            f"AutoRAG pipeline run {run_id} state is '{run_state}' after upgrade, expected SUCCEEDED"
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["autorag_run_accessible"])
    def test_autorag_experiment_details_intact(
        self,
        upgrade_dspa_api_url: str,
        upgrade_dspa_auth_headers: dict[str, str],
        upgrade_dspa_ca_bundle_file: str,
        autorag_upgrade_baseline: dict,
    ) -> None:
        """Verify the run details (display name, pipeline reference, parameters) are intact."""
        run_id = autorag_upgrade_baseline["run_id"]
        expected_display_name = autorag_upgrade_baseline["run_display_name"]

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
    @pytest.mark.dependency(depends=["autorag_run_accessible"])
    def test_autorag_workflow_survived(
        self,
        admin_client: DynamicClient,
        upgrade_namespace: Namespace,
        autorag_upgrade_baseline: dict,
    ) -> None:
        """Verify the Argo Workflow CRD still exists with Succeeded phase."""
        run_id = autorag_upgrade_baseline["run_id"]

        phase = get_workflow_phase(
            admin_client=admin_client,
            namespace=upgrade_namespace.name,
            run_id=run_id,
        )

        assert phase == WORKFLOW_SUCCEEDED, (
            f"Argo Workflow for run {run_id} has phase '{phase}' after upgrade, expected '{WORKFLOW_SUCCEEDED}'"
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["autorag_run_accessible"])
    def test_autorag_artifacts_survived(
        self,
        admin_client: DynamicClient,
        upgrade_namespace: Namespace,
        autorag_upgrade_baseline: dict,
    ) -> None:
        """Verify the workflow execution nodes survived the upgrade."""
        run_id = autorag_upgrade_baseline["run_id"]

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
    def test_autorag_managed_pipeline_accessible(
        self,
        upgrade_managed_pipeline: dict[str, str] | None,
    ) -> None:
        """Verify the managed AutoRAG pipeline is still discoverable after upgrade."""
        assert upgrade_managed_pipeline is not None, "Managed AutoRAG pipeline not found after upgrade"
        assert upgrade_managed_pipeline.get("pipeline_id"), "Managed pipeline has no pipeline_id after upgrade"
        assert upgrade_managed_pipeline.get("pipeline_version_id"), (
            "Managed pipeline has no pipeline_version_id after upgrade"
        )

    @pytest.mark.post_upgrade
    def test_autorag_pattern_query_after_upgrade(
        self,
        upgrade_ogx_client: OgxClient,
        upgrade_discovered_models: tuple[str, str],
        autorag_upgrade_baseline: dict,
    ) -> None:
        """Verify the AutoRAG pattern can still query OGX via file_search after upgrade."""
        _, generation_model = upgrade_discovered_models
        vector_store_ids = autorag_upgrade_baseline.get("vector_store_ids", [])

        assert vector_store_ids, "No vector_store_ids found in upgrade baseline — cannot verify RAG query"

        vector_store = upgrade_ogx_client.vector_stores.retrieve(vector_store_id=vector_store_ids[0])
        assert_rag_query_works(
            ogx_client=upgrade_ogx_client,
            model_id=generation_model,
            vector_store=vector_store,
        )
