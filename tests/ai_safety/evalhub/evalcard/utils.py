from __future__ import annotations

import requests

from tests.ai_safety.evalhub.evalcard.constants import (
    EVALCARD_BENCHMARK_RESULT_REQUIRED_KEYS,
    EVALCARD_CARD_VERSION,
    EVALCARD_CONTEXT_REQUIRED_KEYS,
    EVALCARD_METADATA_REQUIRED_KEYS,
    EVALCARD_REQUIRED_TOP_LEVEL_KEYS,
    EVALCARD_RESULTS_REQUIRED_KEYS,
    EVALCARD_SCHEMA_VERSION,
    EVALHUB_EVALCARD_PATH_TEMPLATE,
)
from tests.ai_safety.evalhub.utils import build_headers


def get_evalcard_http(
    host: str,
    token: str,
    ca_bundle_file: str,
    tenant: str,
    job_id: str,
) -> requests.Response:
    """GET the eval card for a job via the discovery API."""
    path = EVALHUB_EVALCARD_PATH_TEMPLATE.format(job_id=job_id)
    url = f"https://{host}{path}"
    return requests.get(
        url=url,
        headers=build_headers(token=token, tenant=tenant),
        verify=ca_bundle_file,
        timeout=30,
    )


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
    assert isinstance(context["benchmarks"], list), "context.benchmarks must be a list"
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
