from __future__ import annotations

import requests
import structlog
from timeout_sampler import TimeoutSampler

from tests.ai_safety.evalhub.evalcard.constants import (
    EVALCARD_BENCHMARK_RESULT_REQUIRED_KEYS,
    EVALCARD_CARD_VERSION,
    EVALCARD_CONTEXT_REQUIRED_KEYS,
    EVALCARD_METADATA_REQUIRED_KEYS,
    EVALCARD_MLFLOW_ARTIFACT_NAME,
    EVALCARD_MLFLOW_RUN_TAG_KEY,
    EVALCARD_REQUIRED_TOP_LEVEL_KEYS,
    EVALCARD_RESULTS_REQUIRED_KEYS,
    EVALCARD_SCHEMA_VERSION,
    MLFLOW_ARTIFACTS_PATH,
    MLFLOW_RUNS_SEARCH_PATH,
    MLFLOW_WORKSPACE_HEADER,
)
from utilities.guardrails import get_auth_headers
from utilities.registry_utils import pull_manifest_from_oci_registry

LOGGER = structlog.get_logger(name=__name__)


def _mlflow_headers(token: str, workspace: str) -> dict[str, str]:
    headers = get_auth_headers(token=token)
    headers[MLFLOW_WORKSPACE_HEADER] = workspace
    return headers


def find_evalcard_mlflow_run_id(
    mlflow_url: str,
    token: str,
    ca_bundle_file: str,
    workspace: str,
    experiment_id: str,
    job_id: str,
) -> str | None:
    """Return the ID of the MLflow run EvalHub created for the job's eval card, or None if not found yet."""
    resp = requests.post(
        url=f"{mlflow_url}{MLFLOW_RUNS_SEARCH_PATH}",
        headers=_mlflow_headers(token=token, workspace=workspace),
        json={
            "experiment_ids": [experiment_id],
            "filter": f"tags.{EVALCARD_MLFLOW_RUN_TAG_KEY} = '{job_id}'",
            "max_results": 10,
        },
        verify=ca_bundle_file,
        timeout=30,
    )
    assert resp.status_code == 200, f"MLflow runs/search failed with {resp.status_code}: {resp.text}"
    runs = resp.json().get("runs", [])
    if not runs:
        return None
    return runs[0]["info"]["run_id"]


def get_evalcard_from_mlflow(
    mlflow_url: str,
    token: str,
    ca_bundle_file: str,
    workspace: str,
    experiment_id: str,
    job_id: str,
    timeout: int = 120,
    sleep: int = 5,
) -> dict:
    """Fetch the eval card JSON that EvalHub uploaded to MLflow for a job.

    EvalHub uploads the card after the job reaches a terminal state, and MLflow's tag
    index may lag behind, so the run lookup is retried.
    """
    run_id = None
    for run_id in TimeoutSampler(
        wait_timeout=timeout,
        sleep=sleep,
        func=find_evalcard_mlflow_run_id,
        mlflow_url=mlflow_url,
        token=token,
        ca_bundle_file=ca_bundle_file,
        workspace=workspace,
        experiment_id=experiment_id,
        job_id=job_id,
    ):
        if run_id:
            break
    LOGGER.info(f"Found eval card MLflow run {run_id} for job {job_id} in experiment {experiment_id}")

    artifact_path = f"{experiment_id}/{run_id}/artifacts/{EVALCARD_MLFLOW_ARTIFACT_NAME}"
    resp = requests.get(
        url=f"{mlflow_url}{MLFLOW_ARTIFACTS_PATH}/{artifact_path}",
        headers=_mlflow_headers(token=token, workspace=workspace),
        verify=ca_bundle_file,
        timeout=30,
    )
    assert resp.status_code == 200, (
        f"Failed to download MLflow artifact {artifact_path}, got {resp.status_code}: {resp.text}"
    )
    return resp.json()


def _oci_manifest_status_code(registry_url: str, repo: str, tag: str) -> int:
    resp = requests.get(
        url=f"{registry_url}/v2/{repo}/manifests/{tag}",
        headers={"Accept": "application/vnd.oci.image.manifest.v1+json"},
        timeout=10,
    )
    return resp.status_code


def get_evalcard_manifest_from_oci(
    registry_url: str,
    repo: str,
    tag: str,
    timeout: int = 120,
    sleep: int = 5,
) -> dict:
    """Fetch the eval card manifest EvalHub pushed to an OCI registry.

    EvalHub pushes the card after the job reaches a terminal state, so the manifest
    lookup is retried until it appears.
    """
    for status_code in TimeoutSampler(
        wait_timeout=timeout,
        sleep=sleep,
        func=_oci_manifest_status_code,
        registry_url=registry_url,
        repo=repo,
        tag=tag,
    ):
        if status_code == 200:
            break
        LOGGER.info(f"OCI manifest {repo}:{tag} not available yet (status {status_code})")
    return pull_manifest_from_oci_registry(registry_url=registry_url, repo=repo, tag=tag)


def validate_evalcard_schema(card: dict, job_id: str) -> None:
    """Assert required keys and version strings in an eval card dict."""
    for key in EVALCARD_REQUIRED_TOP_LEVEL_KEYS:
        assert key in card, f"EvalCard missing required top-level key: {key}"

    assert card["schema_version"] == EVALCARD_SCHEMA_VERSION, (
        f"Expected schema_version={EVALCARD_SCHEMA_VERSION}, got {card['schema_version']}"
    )
    assert card["card_version"] == EVALCARD_CARD_VERSION, (
        f"Expected card_version={EVALCARD_CARD_VERSION}, got {card['card_version']}"
    )

    metadata = card["metadata"]
    for key in EVALCARD_METADATA_REQUIRED_KEYS:
        assert key in metadata, f"EvalCard metadata missing key: {key}"
    assert metadata["evaluation_job_id"] == job_id, (
        f"Expected evaluation_job_id={job_id}, got {metadata['evaluation_job_id']}"
    )
    assert metadata["created_at"], "created_at must not be empty"

    context = card["context"]
    for key in EVALCARD_CONTEXT_REQUIRED_KEYS:
        assert key in context, f"EvalCard context missing key: {key}"
    assert context["model"], "model must not be empty"
    if not context.get("collection_id"):
        assert isinstance(context.get("benchmarks"), list), "context.benchmarks must be a list"
        assert len(context["benchmarks"]) >= 1, "context.benchmarks must have at least one entry"

    results = card["results"]
    for key in EVALCARD_RESULTS_REQUIRED_KEYS:
        assert key in results, f"EvalCard results missing key: {key}"
    assert isinstance(results["benchmarks"], list), "results.benchmarks must be a list"
    assert len(results["benchmarks"]) >= 1, "results.benchmarks must have at least one entry"

    for bench in results["benchmarks"]:
        for key in EVALCARD_BENCHMARK_RESULT_REQUIRED_KEYS:
            assert key in bench, f"Benchmark result missing key: {key}"


def validate_evalcard_complete_mode(card: dict) -> None:
    """Assert that a complete-mode card has per-benchmark breakdowns and thresholds."""
    for bench in card["results"]["benchmarks"]:
        assert "test" in bench, "Complete card benchmark must have 'test' block"
        test_block = bench["test"]
        assert "primary_score" in test_block, "Complete card test must have primary_score"
        assert "threshold" in test_block, "Complete card test must have threshold"
        assert "pass" in test_block, "Complete card test must have pass boolean"


def validate_evalcard_collection_fields(card: dict, collection_id: str) -> None:
    """Assert collection-specific fields in an eval card."""
    context = card["context"]
    assert context.get("collection_id") == collection_id, (
        f"Expected collection_id={collection_id}, got {context.get('collection_id')}"
    )

    results = card["results"]
    assert "collection" in results, "Collection eval card must have 'collection' in results"
    collection_result = results["collection"]
    assert "test" in collection_result, "Collection results must have 'test' block"
    test_block = collection_result["test"]
    assert "score" in test_block, "Collection test must have score"
    assert "threshold" in test_block, "Collection test must have threshold"
    assert "pass" in test_block, "Collection test must have pass boolean"
