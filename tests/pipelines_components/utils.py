import tempfile
from pathlib import Path
from typing import Any

import requests
import structlog
from kubernetes.client.rest import ApiException
from kubernetes.dynamic import DynamicClient
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.pipelines_components.constants import PIPELINE_POLL_INTERVAL
from utilities.resources.workflow import Workflow

LOGGER = structlog.get_logger(name=__name__)

WORKFLOW_SUCCEEDED: str = "Succeeded"
WORKFLOW_TERMINAL_PHASES: set[str] = {"Succeeded", "Failed", "Error"}


def resolve_pipeline_yaml(value: str) -> str:
    """Resolve a pipeline YAML value to a local file path.

    If the value is a URL (https://), downloads the file to a temp location.
    If it's a local path, validates the file exists.

    Returns:
        Absolute path to the pipeline YAML file.

    Raises:
        FileNotFoundError: If the local path does not exist or the download fails.
    """
    if value.startswith(("https://", "http://")):
        LOGGER.info(f"Downloading pipeline YAML from {value}")
        resp = requests.get(url=value, timeout=60)
        resp.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp.write(resp.content)
        LOGGER.info(f"Pipeline YAML downloaded to {tmp.name}")
        return tmp.name

    path = Path(value)  # noqa: FCN001
    if not path.is_file():
        raise FileNotFoundError(
            f"Pipeline YAML not found: {value!r}\n"
            f"Provide a local file path or a URL (https://...) to a compiled pipeline YAML."
        )
    return str(path.resolve())


def _raise_for_status(resp: requests.Response) -> None:
    """Raise on HTTP errors, including the server response body in the message."""
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        raise requests.HTTPError(
            f"{exc} — Response body: {resp.text}",
            response=resp,
        ) from exc


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
    _raise_for_status(resp=resp)
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
    _raise_for_status(resp=resp)
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
            exceptions_dict={ApiException: [], ConnectionError: [], TimeoutError: []},
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


def delete_pipeline(
    api_url: str,
    headers: dict[str, str],
    pipeline_id: str,
    ca_bundle: str,
) -> None:
    """Delete a pipeline from the DSPA."""
    resp = requests.delete(
        url=f"{api_url}/apis/v2beta1/pipelines/{pipeline_id}",
        headers=headers,
        verify=ca_bundle,
        timeout=60,
    )
    if resp.ok:
        LOGGER.info(f"Deleted pipeline {pipeline_id}")
    else:
        LOGGER.warning(f"Failed to delete pipeline {pipeline_id}: {resp.status_code} {resp.text}")


def delete_pipeline_run(
    api_url: str,
    headers: dict[str, str],
    run_id: str,
    ca_bundle: str,
) -> None:
    """Delete a pipeline run from the DSPA."""
    resp = requests.delete(
        url=f"{api_url}/apis/v2beta1/runs/{run_id}",
        headers=headers,
        verify=ca_bundle,
        timeout=60,
    )
    if resp.ok:
        LOGGER.info(f"Deleted pipeline run {run_id}")
    else:
        LOGGER.warning(f"Failed to delete pipeline run {run_id}: {resp.status_code} {resp.text}")


def collect_pipeline_pod_logs(
    admin_client: DynamicClient,
    namespace: str,
    run_id: str,
) -> None:
    """Log failed workflow node messages for post-failure debugging."""
    found = False
    for workflow in Workflow.get(client=admin_client, namespace=namespace):
        labels = workflow.instance.metadata.get("labels", {})
        if labels.get("pipeline/runid") != run_id:
            continue

        found = True
        nodes = workflow.instance.get("status", {}).get("nodes", {})
        for node_name, node in nodes.items():
            node_phase = node.get("phase", "")
            if node_phase in ("Failed", "Error"):
                message = node.get("message", "<no message>")
                display_name = node.get("displayName", node_name)
                LOGGER.error(f"Workflow node '{display_name}' {node_phase}: {message}")

    if not found:
        LOGGER.warning(f"No Argo Workflow found for pipeline run {run_id} in namespace {namespace}")
