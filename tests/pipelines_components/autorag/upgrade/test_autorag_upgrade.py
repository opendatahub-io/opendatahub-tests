"""Check that a managed AutoRAG run survives a product upgrade."""

from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace

from tests.pipelines_components.constants import AUTORAG_INPUT_DATA_KEY, AUTORAG_PIPELINE_START_TIMEOUT
from tests.pipelines_components.upgrade_utils import (
    load_upgrade_baseline,
    save_upgrade_baseline,
    wait_for_upgrade_run_start,
)
from tests.pipelines_components.utils import get_pipeline_run
from utilities.resources.workflow import Workflow

BASELINE_NAME = "autorag-upgrade-baseline"


class TestAutoRAGUpgrade:
    """Verify startup before upgrade and persistence of the same run afterward."""

    @pytest.mark.pre_upgrade
    def test_autorag_run_starts_before_upgrade(
        self,
        admin_client: DynamicClient,
        autorag_upgrade_namespace: Namespace,
        autorag_upgrade_api_url: str,
        autorag_upgrade_auth_headers: dict[str, str],
        autorag_upgrade_ca_bundle: str,
        autorag_upgrade_pipeline: dict[str, str],
        autorag_upgrade_run_id: str,
    ) -> None:
        """Given local documents and placeholder connections, when a managed run starts, then record its identity."""
        run = get_pipeline_run(
            api_url=autorag_upgrade_api_url,
            headers=autorag_upgrade_auth_headers,
            run_id=autorag_upgrade_run_id,
            ca_bundle=autorag_upgrade_ca_bundle,
        )
        reference = run.get("pipeline_version_reference", {})
        assert run["run_id"] == autorag_upgrade_run_id
        assert reference.get("pipeline_id") == autorag_upgrade_pipeline["pipeline_id"]
        assert reference.get("pipeline_version_id") == autorag_upgrade_pipeline["pipeline_version_id"]
        assert run.get("runtime_config", {}).get("parameters", {}).get("input_data_keys") == [AUTORAG_INPUT_DATA_KEY]
        workflow_name = wait_for_upgrade_run_start(
            client=admin_client,
            namespace=autorag_upgrade_namespace.name,
            run_id=autorag_upgrade_run_id,
            timeout=AUTORAG_PIPELINE_START_TIMEOUT,
        )
        save_upgrade_baseline(
            client=admin_client,
            namespace=autorag_upgrade_namespace.name,
            name=BASELINE_NAME,
            baseline={
                "run_id": autorag_upgrade_run_id,
                "display_name": run["display_name"],
                "pipeline_version_reference": reference,
                "input_data_keys": [AUTORAG_INPUT_DATA_KEY],
                "workflow_name": workflow_name,
            },
        )

    @pytest.mark.post_upgrade
    def test_autorag_run_survives_upgrade(
        self,
        admin_client: DynamicClient,
        autorag_upgrade_namespace: Namespace,
        autorag_upgrade_api_url: str,
        autorag_upgrade_auth_headers: dict[str, str],
        autorag_upgrade_ca_bundle: str,
        autorag_upgrade_pipeline: dict[str, str],
    ) -> None:
        """Given a pre-upgrade run, when the cluster upgrades, then its KFP and workflow records survive."""
        baseline: dict[str, Any] = load_upgrade_baseline(
            client=admin_client, namespace=autorag_upgrade_namespace.name, name=BASELINE_NAME
        )
        run = get_pipeline_run(
            api_url=autorag_upgrade_api_url,
            headers=autorag_upgrade_auth_headers,
            run_id=baseline["run_id"],
            ca_bundle=autorag_upgrade_ca_bundle,
        )
        assert run["run_id"] == baseline["run_id"]
        assert run["display_name"] == baseline["display_name"]
        assert run["pipeline_version_reference"] == baseline["pipeline_version_reference"]
        assert run["runtime_config"]["parameters"]["input_data_keys"] == baseline["input_data_keys"]
        assert autorag_upgrade_pipeline["pipeline_id"] == baseline["pipeline_version_reference"]["pipeline_id"], (
            "Original managed AutoRAG pipeline is no longer discoverable"
        )
        workflows = list(
            Workflow.get(
                client=admin_client,
                namespace=autorag_upgrade_namespace.name,
                label_selector=f"pipeline/runid={baseline['run_id']}",
            )
        )
        assert len(workflows) == 1 and workflows[0].name == baseline["workflow_name"], (
            f"Workflow {baseline['workflow_name']} for run {baseline['run_id']} did not survive the upgrade"
        )
