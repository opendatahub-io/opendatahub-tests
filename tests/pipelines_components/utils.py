import json
from typing import Any

import requests
import structlog
from kubernetes.dynamic import DynamicClient
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from utilities.resources.workflow import Workflow

LOGGER = structlog.get_logger(name=__name__)

WORKFLOW_SUCCEEDED: str = "Succeeded"


def _raise_for_status(resp: requests.Response) -> None:
    """Raise on HTTP errors, including the server response body in the message."""
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        raise requests.HTTPError(
            f"{exc} — Response body: {resp.text}",
            response=resp,
        ) from exc


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


def get_pipeline_run(
    api_url: str,
    headers: dict[str, str],
    run_id: str,
    ca_bundle: str,
) -> dict:
    """Get a pipeline run by ID from the DSPA."""
    resp = requests.get(
        url=f"{api_url}/apis/v2beta1/runs/{run_id}",
        headers=headers,
        verify=ca_bundle,
        timeout=60,
    )
    _raise_for_status(resp=resp)
    return resp.json()


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
