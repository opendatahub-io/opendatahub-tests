"""Persist pipeline run identities and verify startup across an upgrade boundary."""

from typing import Any

import yaml
from kubernetes.client.rest import ApiException
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.pipelines_components.constants import PIPELINE_POLL_INTERVAL
from tests.pipelines_components.utils import WORKFLOW_SUCCEEDED, collect_pipeline_pod_logs, get_workflow_phase
from utilities.resources.workflow import Workflow


def save_upgrade_baseline(client: DynamicClient, namespace: str | None, name: str, baseline: dict[str, Any]) -> None:
    """Persist a verified run's identifiers for the post-upgrade phase.

    Args:
        client: Kubernetes client.
        namespace: Namespace kept across the upgrade.
        name: Unique ConfigMap name for this run.
        baseline: Run and workflow identifiers to verify later.
    """
    assert namespace is not None, "Upgrade namespace must have a name"
    with ConfigMap(
        client=client,
        name=name,
        namespace=namespace,
        data={"baselines.yaml": yaml.safe_dump(baseline)},
        teardown=False,
    ):
        pass


def load_upgrade_baseline(client: DynamicClient, namespace: str | None, name: str) -> dict[str, Any]:
    """Read the baseline saved before the upgrade.

    Args:
        client: Kubernetes client.
        namespace: Namespace kept across the upgrade.
        name: ConfigMap name used during pre-upgrade.

    Returns:
        Stored run and workflow identifiers.
    """
    assert namespace is not None, "Upgrade namespace must have a name"
    config_map = ConfigMap(client=client, name=name, namespace=namespace, ensure_exists=True)
    return yaml.safe_load(config_map.instance.data["baselines.yaml"])


def wait_for_upgrade_run_start(client: DynamicClient, namespace: str | None, run_id: str, timeout: int) -> str:
    """Wait for a run to start and return the Argo Workflow's name.

    Args:
        client: Kubernetes client.
        namespace: DSPA namespace.
        run_id: Submitted KFP run ID.
        timeout: Maximum seconds to wait for workflow progression.

    Returns:
        The name of the workflow created for the run.

    Raises:
        AssertionError: If the workflow fails or never starts.
    """
    assert namespace is not None, "Upgrade namespace must have a name"
    last_phase: str | None = None
    try:
        for phase in TimeoutSampler(
            wait_timeout=timeout,
            sleep=PIPELINE_POLL_INTERVAL,
            func=get_workflow_phase,
            exceptions_dict={ApiException: [], ConnectionError: [], TimeoutError: []},
            admin_client=client,
            namespace=namespace,
            run_id=run_id,
        ):
            last_phase = phase
            if phase in ("Running", WORKFLOW_SUCCEEDED):
                workflows = list(
                    Workflow.get(client=client, namespace=namespace, label_selector=f"pipeline/runid={run_id}")
                )
                assert len(workflows) == 1, f"Expected one workflow for run {run_id}, got {len(workflows)}"
                return workflows[0].name
            if phase in ("Failed", "Error"):
                collect_pipeline_pod_logs(admin_client=client, namespace=namespace, run_id=run_id)
                raise AssertionError(f"Run {run_id} failed before progressing: {phase}")
    except TimeoutExpiredError as error:
        raise AssertionError(f"Run {run_id} did not start within {timeout}s (last phase: {last_phase})") from error
    raise AssertionError(f"Run {run_id} stopped being polled in phase {last_phase}")
