from typing import Any

import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.resource import NamespacedResource
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.pipelines_components.constants import PIPELINE_POLL_INTERVAL

LOGGER = structlog.get_logger(name=__name__)

WORKFLOW_SUCCEEDED: str = "Succeeded"
WORKFLOW_TERMINAL_PHASES: set[str] = {"Succeeded", "Failed", "Error"}


class Workflow(NamespacedResource):
    """Argo Workflow resource used by Data Science Pipelines."""

    api_group = "argoproj.io"
    api_version = "v1alpha1"


def upload_pipeline(
    api_url: str,
    headers: dict[str, str],
    pipeline_yaml_path: str,
    pipeline_name: str,
    ca_bundle: str,
) -> str:
    """Upload a compiled pipeline YAML to the DSPA and return the pipeline ID."""
    with open(pipeline_yaml_path, "rb") as yaml_file:
        resp = requests.post(
            url=f"{api_url}/apis/v2beta1/pipelines/upload",
            headers=headers,
            files={"uploadfile": (f"{pipeline_name}.yaml", yaml_file, "application/x-yaml")},
            params={"name": pipeline_name},
            verify=ca_bundle,
            timeout=60,
        )
    resp.raise_for_status()
    return resp.json()["pipeline_id"]


def create_pipeline_run(
    api_url: str,
    headers: dict[str, str],
    pipeline_id: str,
    run_name: str,
    parameters: dict[str, Any],
    ca_bundle: str,
) -> str:
    """Create a pipeline run and return the run ID."""
    resp = requests.post(
        url=f"{api_url}/apis/v2beta1/runs",
        headers=headers,
        json={
            "display_name": run_name,
            "pipeline_version_reference": {"pipeline_id": pipeline_id},
            "runtime_config": {"parameters": parameters},
        },
        verify=ca_bundle,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["run_id"]


def get_workflow_phase(
    admin_client: DynamicClient,
    namespace: str,
    run_id: str,
) -> str | None:
    """Get the phase of the Argo Workflow associated with a pipeline run ID."""
    for workflow in Workflow.get(client=admin_client, namespace=namespace):
        labels = workflow.instance.metadata.get("labels", {})
        if labels.get("pipeline/runid") == run_id:
            return workflow.instance.get("status", {}).get("phase")
    return None


def wait_for_pipeline_run(
    admin_client: DynamicClient,
    namespace: str,
    run_id: str,
    timeout: int,
) -> str:
    """Poll the Argo Workflow until it reaches a terminal phase. Returns the phase string."""
    LOGGER.info(f"Waiting for pipeline run {run_id} (timeout={timeout}s)")

    try:
        for phase in TimeoutSampler(
            wait_timeout=timeout,
            sleep=PIPELINE_POLL_INTERVAL,
            func=get_workflow_phase,
            admin_client=admin_client,
            namespace=namespace,
            run_id=run_id,
        ):
            LOGGER.info(f"Pipeline run {run_id}: {phase}")
            if phase and phase in WORKFLOW_TERMINAL_PHASES:
                return phase
    except TimeoutExpiredError as err:
        msg = f"Pipeline run {run_id} did not complete within {timeout}s"
        LOGGER.error(msg)
        raise TimeoutError(msg) from err

    msg = f"Pipeline run {run_id} exited polling without reaching terminal state"
    raise RuntimeError(msg)
