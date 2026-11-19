"""Helpers for IBM CLEAR EvalHub tests (RHOAIENG-63558)."""

from __future__ import annotations

import json
import uuid
from typing import Any

import requests
import structlog

from tests.ai_safety.evalhub.clear.constants import (
    CLEAR_ADAPTER_ENTRYPOINT,
    CLEAR_ADAPTER_IMAGE,
    CLEAR_BENCHMARK_ID,
    CLEAR_EXPECTED_METRICS,
    CLEAR_TRACE_FILENAME,
    IBM_CLEAR_PROVIDER_ID,
    IBM_CLEAR_PROVIDER_NAME,
    MLFLOW_EXPERIMENTS_CREATE_PATH,
    MLFLOW_EXPERIMENTS_GET_BY_NAME_PATH,
    MLFLOW_TRACES_SEARCH_PATH,
    MLFLOW_TRACES_V3_PATH,
    MLFLOW_WORKSPACE_HEADER,
)
from tests.ai_safety.evalhub.constants import (
    EVALHUB_PROVIDERS_PATH,
    EVALHUB_VLLM_EMULATOR_PORT,
)
from tests.ai_safety.evalhub.utils import build_headers
from utilities.guardrails import get_auth_headers

LOGGER = structlog.get_logger(name=__name__)


def build_minimal_agent_trace(
    trace_id: str | None = None,
    *,
    user_query: str = "What is 2+2?",
    agent_answer: str = "4",
) -> dict[str, Any]:
    """Build a compact CLEAR-compatible agent trace (LangGraph / MLflow-style spans).

    Args:
        trace_id: Optional stable trace id; a random id is generated when omitted.
        user_query: Input recorded on the root agent span.
        agent_answer: Output recorded on the root and LLM spans.

    Returns:
        Trace document with top-level ``spans`` matching CLEAR sample layout.
    """
    request_id = trace_id or f"tr-fvt-{uuid.uuid4().hex[:16]}"
    root_span_id = uuid.uuid4().hex[:16]
    llm_span_id = uuid.uuid4().hex[:16]
    start_ns = 1_700_000_000_000_000_000
    mid_ns = start_ns + 50_000_000
    end_ns = start_ns + 100_000_000

    return {
        "trace_id": request_id,
        "state": "OK",
        "execution_time_ms": 100,
        "timestamp_ms": start_ns // 1_000_000,
        "request_preview": user_query,
        "response_preview": agent_answer,
        "tags": {
            "mlflow.traceName": "fvt-minimal-agent",
            "evalhub.fvt": "clear-mlflow-traces",
        },
        "spans": [
            {
                "name": "agent",
                "span_id": root_span_id,
                "parent_span_id": None,
                "start_time_unix_nano": start_ns,
                "end_time_unix_nano": end_ns,
                "duration_ms": 100,
                "status": {"status_code": "OK", "description": ""},
                "inputs": {"query": user_query},
                "outputs": {"answer": agent_answer},
                "attributes": {
                    "mlflow.spanType": "AGENT",
                    "mlflow.traceRequestId": request_id,
                },
            },
            {
                "name": "chat_model",
                "span_id": llm_span_id,
                "parent_span_id": root_span_id,
                "start_time_unix_nano": mid_ns,
                "end_time_unix_nano": end_ns,
                "duration_ms": 50,
                "status": {"status_code": "OK", "description": ""},
                "inputs": {
                    "messages": [{"role": "user", "content": user_query}],
                },
                "outputs": {
                    "message": {"role": "assistant", "content": agent_answer},
                },
                "attributes": {
                    "mlflow.spanType": "LLM",
                    "mlflow.traceRequestId": request_id,
                },
            },
        ],
    }


def build_clear_provider_payload(
    *,
    name: str = IBM_CLEAR_PROVIDER_NAME,
    image: str = CLEAR_ADAPTER_IMAGE,
) -> dict[str, Any]:
    """Build a user-provider create payload for the CLEAR adapter image."""
    return {
        "name": name,
        "description": "FVT provider for IBM CLEAR direct MLflow trace ingestion (RHOAIENG-63558)",
        "benchmarks": [
            {"id": CLEAR_BENCHMARK_ID, "name": "Agentic Evaluation"},
        ],
        "runtime": {
            "k8s": {
                "image": image,
                "entrypoint": CLEAR_ADAPTER_ENTRYPOINT,
                "cpu_request": "100m",
                "memory_request": "256Mi",
                "cpu_limit": "1",
                "memory_limit": "2Gi",
            },
        },
    }


def build_clear_benchmark(
    provider_id: str,
    *,
    benchmark_id: str = CLEAR_BENCHMARK_ID,
    parameters: dict[str, Any] | None = None,
    test_data_ref: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a CLEAR benchmark entry for a job submission payload.

    Args:
        provider_id: Provider id (``ibm-clear`` or a user-created provider id).
        benchmark_id: CLEAR benchmark id.
        parameters: Extra/override CLEAR parameters.
        test_data_ref: Optional staged-data reference (pvc/s3/git/hf).

    Returns:
        Benchmark dict suitable for ``JobSubmissionRequest.benchmarks``.
    """
    params: dict[str, Any] = {
        "eval_model_name": "emulatedModel",
        "provider": "openai",
        "agent_framework": "langgraph",
        "observability_framework": "mlflow",
        "inference_backend": "litellm",
        "separate_tools": False,
    }
    if parameters:
        params.update(parameters)

    benchmark: dict[str, Any] = {
        "id": benchmark_id,
        "provider_id": provider_id,
        "parameters": params,
    }
    if test_data_ref is not None:
        benchmark["test_data_ref"] = test_data_ref
    return benchmark


def build_clear_job_payload(
    *,
    model_service_name: str,
    tenant_namespace: str,
    provider_id: str,
    job_name: str,
    parameters: dict[str, Any] | None = None,
    test_data_ref: dict[str, Any] | None = None,
    experiment_name: str | None = None,
    model_name: str = "emulatedModel",
) -> dict[str, Any]:
    """Build an EvalHub job payload targeting the CLEAR adapter.

    Args:
        model_service_name: Kubernetes Service name for the judge LLM emulator.
        tenant_namespace: Tenant namespace hosting the emulator Service.
        provider_id: CLEAR provider id.
        job_name: Job display name.
        parameters: CLEAR benchmark parameters (mlflow_traces_*, etc.).
        test_data_ref: Optional staged-data reference for file traces.
        experiment_name: Optional MLflow experiment for CLEAR result upload.
        model_name: Judge model name passed in ``model.name``.

    Returns:
        Job request body dict.
    """
    model_url = f"http://{model_service_name}.{tenant_namespace}.svc.cluster.local:{EVALHUB_VLLM_EMULATOR_PORT}/v1"
    payload: dict[str, Any] = {
        "name": job_name,
        "model": {
            "url": model_url,
            "name": model_name,
        },
        "benchmarks": [
            build_clear_benchmark(
                provider_id=provider_id,
                parameters=parameters,
                test_data_ref=test_data_ref,
            ),
        ],
    }
    if experiment_name is not None:
        payload["experiment"] = {"name": experiment_name}
    return payload


def find_provider_id_by_resource_or_name(
    *,
    providers_response: dict[str, Any],
    resource_id: str = IBM_CLEAR_PROVIDER_ID,
    name: str | None = None,
) -> str | None:
    """Return a provider id from a list-providers response, if present."""
    for item in providers_response.get("items", []):
        item_id = item.get("resource", {}).get("id")
        if item_id == resource_id:
            return item_id
        if name is not None and item.get("name") == name:
            return item_id
    return None


def get_provider(
    host: str,
    token: str,
    ca_bundle_file: str,
    tenant: str,
    provider_id: str,
) -> dict[str, Any]:
    """GET a single provider resource."""
    url = f"https://{host}{EVALHUB_PROVIDERS_PATH}/{provider_id}"
    response = requests.get(
        url=url,
        headers=build_headers(token=token, tenant=tenant),
        verify=ca_bundle_file,
        timeout=10,
    )
    assert response.status_code == 200, (
        f"Expected 200 for GET provider {provider_id}, got {response.status_code}: {response.text}"
    )
    return response.json()


def validate_clear_job_completed(job_data: dict[str, Any], *, provider_id: str) -> None:
    """Assert a CLEAR job completed with expected benchmark metrics.

    Args:
        job_data: Terminal job response from ``wait_for_evalhub_job``.
        provider_id: Expected provider id on the benchmark result.

    Raises:
        AssertionError: If state/results/metrics are missing or incorrect.
    """
    state = job_data.get("status", {}).get("state")
    assert state == "completed", (
        f"Expected CLEAR job state 'completed', got '{state}': {job_data.get('status', {}).get('message')}"
    )

    results = job_data.get("results", {})
    benchmarks = results.get("benchmarks", [])
    assert benchmarks, f"CLEAR job completed but has no benchmark results: {results}"

    clear_benches = [benchmark for benchmark in benchmarks if benchmark.get("id") == CLEAR_BENCHMARK_ID]
    assert clear_benches, f"Expected '{CLEAR_BENCHMARK_ID}' in results, got: {[b.get('id') for b in benchmarks]}"
    benchmark = clear_benches[0]
    assert benchmark.get("provider_id") == provider_id, (
        f"Expected provider_id={provider_id!r}, got {benchmark.get('provider_id')!r}"
    )

    metrics = benchmark.get("metrics") or []
    metric_names = {name for metric in metrics if isinstance(metric, dict) and (name := metric.get("name")) is not None}
    missing = [name for name in CLEAR_EXPECTED_METRICS if name not in metric_names]
    assert not missing, f"CLEAR results missing expected metrics {missing}; got {sorted(metric_names)}"


def _mlflow_headers(token: str, workspace: str) -> dict[str, str]:
    headers = get_auth_headers(token=token)
    headers[MLFLOW_WORKSPACE_HEADER] = workspace
    return headers


def ensure_mlflow_experiment(
    mlflow_url: str,
    token: str,
    ca_bundle_file: str,
    workspace: str,
    experiment_name: str,
) -> str:
    """Create or resolve an MLflow experiment by name; return experiment id."""
    headers = _mlflow_headers(token=token, workspace=workspace)
    get_resp = requests.get(
        url=f"{mlflow_url}{MLFLOW_EXPERIMENTS_GET_BY_NAME_PATH}",
        headers=headers,
        params={"experiment_name": experiment_name},
        verify=ca_bundle_file,
        timeout=30,
    )
    if get_resp.status_code == 200:
        experiment_id = get_resp.json().get("experiment", {}).get("experiment_id")
        if experiment_id:
            LOGGER.info("Reusing MLflow experiment", name=experiment_name, experiment_id=experiment_id)
            return experiment_id

    create_resp = requests.post(
        url=f"{mlflow_url}{MLFLOW_EXPERIMENTS_CREATE_PATH}",
        headers=headers,
        json={"name": experiment_name},
        verify=ca_bundle_file,
        timeout=30,
    )
    assert create_resp.status_code == 200, (
        f"MLflow experiments/create failed with {create_resp.status_code}: {create_resp.text}"
    )
    experiment_id = create_resp.json().get("experiment_id")
    assert experiment_id, f"MLflow create response missing experiment_id: {create_resp.text}"
    LOGGER.info("Created MLflow experiment", name=experiment_name, experiment_id=experiment_id)
    return experiment_id


def seed_mlflow_trace_via_rest(
    mlflow_url: str,
    token: str,
    ca_bundle_file: str,
    workspace: str,
    experiment_id: str,
    trace: dict[str, Any] | None = None,
) -> str:
    """Best-effort seed of one agent trace into MLflow via the v3 traces API.

    Creates TraceInfo for ``experiment_id``. Span attachment depends on server
    support; callers should verify with ``search_mlflow_traces`` before relying
    on materialize() seeing span data.

    Args:
        mlflow_url: MLflow base URL including static prefix (e.g. ``.../mlflow``).
        token: Bearer token for MLflow auth.
        ca_bundle_file: CA bundle path for TLS verification.
        workspace: Tenant workspace for ``X-MLFLOW-WORKSPACE``.
        experiment_id: Target MLflow experiment id.
        trace: Optional CLEAR-style trace document; a minimal one is generated.

    Returns:
        Trace / request id that was registered.
    """
    document = trace or build_minimal_agent_trace()
    trace_id = document.get("trace_id") or f"tr-fvt-{uuid.uuid4().hex[:16]}"
    payload = {
        "trace": {
            "trace_info": {
                "trace_id": trace_id,
                "client_request_id": trace_id,
                "state": document.get("state", "OK"),
                "request_preview": document.get("request_preview", ""),
                "response_preview": document.get("response_preview", ""),
                "execution_duration_ms": document.get("execution_time_ms", 100),
                "trace_location": {
                    "type": "MLFLOW_EXPERIMENT",
                    "mlflow_experiment": {"experiment_id": experiment_id},
                },
                "tags": document.get("tags") or {"evalhub.fvt": "clear-mlflow-traces"},
            },
        },
    }
    response = requests.post(
        url=f"{mlflow_url}{MLFLOW_TRACES_V3_PATH}",
        headers=_mlflow_headers(token=token, workspace=workspace),
        json=payload,
        verify=ca_bundle_file,
        timeout=30,
    )
    assert response.status_code in {200, 201}, (
        f"MLflow traces create failed with {response.status_code}: {response.text}"
    )
    LOGGER.info("Seeded MLflow trace", trace_id=trace_id, experiment_id=experiment_id)
    return trace_id


def search_mlflow_traces(
    mlflow_url: str,
    token: str,
    ca_bundle_file: str,
    workspace: str,
    experiment_id: str,
    *,
    max_results: int = 50,
) -> list[dict[str, Any]]:
    """Search MLflow traces for an experiment (2.0 search API)."""
    response = requests.post(
        url=f"{mlflow_url}{MLFLOW_TRACES_SEARCH_PATH}",
        headers=_mlflow_headers(token=token, workspace=workspace),
        json={
            "experiment_ids": [experiment_id],
            "max_results": max_results,
        },
        verify=ca_bundle_file,
        timeout=30,
    )
    assert response.status_code == 200, f"MLflow traces/search failed with {response.status_code}: {response.text}"
    body = response.json()
    return body.get("traces") or body.get("trace_infos") or []


def minimal_trace_json_bytes() -> bytes:
    """Serialize the default minimal agent trace for PVC/ConfigMap seeding."""
    document = build_minimal_agent_trace(trace_id=CLEAR_TRACE_FILENAME.removesuffix(".json"))
    return json.dumps(document, indent=2).encode("utf-8")
