"""Check that managed AutoML runs submitted before an upgrade remain accessible afterward."""

from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace

from tests.pipelines_components.constants import AUTOML_PIPELINE_START_TIMEOUT
from tests.pipelines_components.upgrade_utils import (
    load_upgrade_baseline,
    save_upgrade_baseline,
    wait_for_upgrade_run_start,
)
from tests.pipelines_components.utils import get_pipeline_run
from utilities.resources.workflow import Workflow


@pytest.mark.parametrize(
    "task_type",
    ["regression", "classification", "multiclass", "timeseries"],
    ids=["test_regression", "test_classification", "test_multiclass", "test_timeseries"],
)
class TestAutoMLUpgrade:
    """Verify startup before upgrade and persistence of the same run afterward."""

    @pytest.mark.pre_upgrade
    def test_automl_run_starts_before_upgrade(
        self,
        task_type: str,
        admin_client: DynamicClient,
        upgrade_namespace: Namespace,
        upgrade_dspa_api_url: str,
        upgrade_dspa_auth_headers: dict[str, str],
        upgrade_dspa_ca_bundle_file: str,
        upgrade_managed_pipeline: dict[str, str],
        upgrade_train_data: str,
        upgrade_run_id: str,
    ) -> None:
        """Given generated data, when a managed run starts, then record its identity for upgrade checks."""
        run = get_pipeline_run(
            api_url=upgrade_dspa_api_url,
            headers=upgrade_dspa_auth_headers,
            run_id=upgrade_run_id,
            ca_bundle=upgrade_dspa_ca_bundle_file,
        )
        reference = run.get("pipeline_version_reference", {})
        assert run["run_id"] == upgrade_run_id
        assert reference.get("pipeline_id") == upgrade_managed_pipeline["pipeline_id"]
        assert reference.get("pipeline_version_id") == upgrade_managed_pipeline["pipeline_version_id"]
        assert run.get("runtime_config", {}).get("parameters", {}).get("train_data_file_key") == upgrade_train_data
        workflow_name = wait_for_upgrade_run_start(
            client=admin_client,
            namespace=upgrade_namespace.name,
            run_id=upgrade_run_id,
            timeout=AUTOML_PIPELINE_START_TIMEOUT,
        )
        save_upgrade_baseline(
            client=admin_client,
            namespace=upgrade_namespace.name,
            name=f"automl-upgrade-{task_type}-baseline",
            baseline={
                "run_id": upgrade_run_id,
                "display_name": run["display_name"],
                "pipeline_version_reference": reference,
                "train_data_file_key": upgrade_train_data,
                "workflow_name": workflow_name,
            },
        )

    @pytest.mark.post_upgrade
    def test_automl_run_survives_upgrade(
        self,
        task_type: str,
        admin_client: DynamicClient,
        upgrade_namespace: Namespace,
        upgrade_dspa_api_url: str,
        upgrade_dspa_auth_headers: dict[str, str],
        upgrade_dspa_ca_bundle_file: str,
        upgrade_managed_pipeline: dict[str, str],
    ) -> None:
        """Given a pre-upgrade run, when the cluster upgrades, then its KFP and workflow records survive."""
        baseline: dict[str, Any] = load_upgrade_baseline(
            client=admin_client,
            namespace=upgrade_namespace.name,
            name=f"automl-upgrade-{task_type}-baseline",
        )
        run = get_pipeline_run(
            api_url=upgrade_dspa_api_url,
            headers=upgrade_dspa_auth_headers,
            run_id=baseline["run_id"],
            ca_bundle=upgrade_dspa_ca_bundle_file,
        )
        assert run["run_id"] == baseline["run_id"]
        assert run["display_name"] == baseline["display_name"]
        assert run["pipeline_version_reference"] == baseline["pipeline_version_reference"]
        assert run["runtime_config"]["parameters"]["train_data_file_key"] == baseline["train_data_file_key"]
        assert upgrade_managed_pipeline["pipeline_id"], "Managed AutoML pipeline is no longer discoverable"
        workflows = list(
            Workflow.get(
                client=admin_client,
                namespace=upgrade_namespace.name,
                label_selector=f"pipeline/runid={baseline['run_id']}",
            )
        )
        assert len(workflows) == 1 and workflows[0].name == baseline["workflow_name"], (
            f"Workflow {baseline['workflow_name']} for run {baseline['run_id']} did not survive the upgrade"
        )
