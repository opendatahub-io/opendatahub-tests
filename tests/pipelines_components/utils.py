import atexit
import json
import os
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
        atexit.register(os.unlink, path=tmp.name)  # noqa: FCN001
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
    workflows = list(Workflow.get(client=admin_client, namespace=namespace, label_selector=f"pipeline/runid={run_id}"))
    if workflows:
        return workflows[0].instance.get("status", {}).get("phase")
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
        raise TimeoutExpiredError(msg) from err

    msg = f"Pipeline run {run_id} exited polling without reaching terminal state"
    raise RuntimeError(msg)


def delete_pipeline(
    api_url: str,
    headers: dict[str, str],
    pipeline_id: str,
    ca_bundle: str,
) -> None:
    """Delete a pipeline and all its versions from the DSPA."""
    resp = requests.delete(
        url=f"{api_url}/apis/v2beta1/pipelines/{pipeline_id}",
        headers=headers,
        params={"cascade": "true"},
        verify=ca_bundle,
        timeout=60,
    )
    _raise_for_status(resp=resp)
    LOGGER.info(f"Deleted pipeline {pipeline_id}")


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
    _raise_for_status(resp=resp)
    LOGGER.info(f"Deleted pipeline run {run_id}")


def collect_pipeline_pod_logs(
    admin_client: DynamicClient,
    namespace: str,
    run_id: str,
) -> None:
    """Log failed workflow node messages for post-failure debugging."""
    workflows = list(Workflow.get(client=admin_client, namespace=namespace, label_selector=f"pipeline/runid={run_id}"))
    if not workflows:
        LOGGER.warning(f"No Argo Workflow found for pipeline run {run_id} in namespace {namespace}")
        return

    for workflow in workflows:
        nodes = workflow.instance.get("status", {}).get("nodes", {})
        for node_name, node in nodes.items():
            node_phase = node.get("phase", "")
            if node_phase in ("Failed", "Error"):
                message = node.get("message", "<no message>")
                display_name = node.get("displayName", node_name)
                LOGGER.error(f"Workflow node '{display_name}' {node_phase}: {message}")


# ---------------------------------------------------------------------------
# Managed pipeline helpers
# ---------------------------------------------------------------------------


def use_managed_pipelines(yaml_env_value: str) -> bool:
    """Return True (managed mode) when the YAML env var is empty, False (legacy) when set."""
    return not bool(yaml_env_value)


def find_pipeline_by_display_name(
    api_url: str,
    headers: dict[str, str],
    display_name: str,
    ca_bundle: str,
) -> dict[str, str] | None:
    """Search KFP for a pipeline by display name.

    Returns {"pipeline_id": ..., "pipeline_version_id": ...} or None.
    """
    resp = requests.get(
        url=f"{api_url}/apis/v2beta1/pipelines",
        headers=headers,
        params={
            "filter": json.dumps({
                "predicates": [
                    {
                        "key": "display_name",
                        "operation": "EQUALS",
                        "string_value": display_name,
                    }
                ]
            })
        },
        verify=ca_bundle,
        timeout=60,
    )
    _raise_for_status(resp=resp)
    pipelines = resp.json().get("pipelines", [])
    if not pipelines:
        return None

    pipeline_id = pipelines[0]["pipeline_id"]

    version_resp = requests.get(
        url=f"{api_url}/apis/v2beta1/pipelines/{pipeline_id}/versions",
        headers=headers,
        params={"sort_by": "created_at desc", "page_size": "1"},
        verify=ca_bundle,
        timeout=60,
    )
    _raise_for_status(resp=version_resp)
    versions = version_resp.json().get("pipeline_versions", [])
    version_id = versions[0]["pipeline_version_id"] if versions else ""

    return {"pipeline_id": pipeline_id, "pipeline_version_id": version_id}


def wait_for_managed_pipeline(
    api_url: str,
    headers: dict[str, str],
    display_name: str,
    ca_bundle: str,
    timeout: int,
    poll_interval: int,
) -> dict[str, str]:
    """Poll KFP until a managed pipeline with the given display name appears."""
    LOGGER.info(f"Waiting for managed pipeline '{display_name}' (timeout={timeout}s)")

    for result in TimeoutSampler(
        wait_timeout=timeout,
        sleep=poll_interval,
        func=find_pipeline_by_display_name,
        exceptions_dict={requests.ConnectionError: [], requests.HTTPError: []},
        api_url=api_url,
        headers=headers,
        display_name=display_name,
        ca_bundle=ca_bundle,
    ):
        if result is not None:
            LOGGER.info(
                f"Found managed pipeline '{display_name}': "
                f"pipeline_id={result['pipeline_id']}, version_id={result['pipeline_version_id']}"
            )
            return result

    raise TimeoutExpiredError(f"Managed pipeline '{display_name}' not found within {timeout}s")


def create_pipeline_run_managed(
    api_url: str,
    headers: dict[str, str],
    pipeline_id: str,
    pipeline_version_id: str,
    run_name: str,
    parameters: dict[str, Any],
    ca_bundle: str,
) -> str:
    """Create a pipeline run for a managed pipeline (with version_id) and return the run ID."""
    resp = requests.post(
        url=f"{api_url}/apis/v2beta1/runs",
        headers=headers,
        json={
            "display_name": run_name,
            "pipeline_version_reference": {
                "pipeline_id": pipeline_id,
                "pipeline_version_id": pipeline_version_id,
            },
            "runtime_config": {"parameters": parameters},
        },
        verify=ca_bundle,
        timeout=60,
    )
    _raise_for_status(resp=resp)
    return resp.json()["run_id"]
