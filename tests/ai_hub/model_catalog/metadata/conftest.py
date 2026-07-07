import random
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.pod import Pod

from tests.ai_hub.constants import CATALOG_CONTAINER
from tests.ai_hub.model_catalog.constants import (
    MODEL_ARTIFACT_TYPE,
    PERFORMANCE_DATA_DIR,
    VALIDATED_CATALOG_ID,
)
from tests.ai_hub.model_catalog.metadata.constants import ALL_ARTIFACT_CATEGORIES
from tests.ai_hub.model_catalog.metadata.utils import get_labels_from_configmaps
from tests.ai_hub.model_catalog.search.utils import fetch_all_artifacts_with_dynamic_paging
from tests.ai_hub.model_catalog.utils import get_models_from_catalog_api
from tests.ai_hub.utils import execute_get_command

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture()
def expected_labels_by_asset_type(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    model_registry_namespace: str,
) -> list[dict[str, Any]]:
    """Get expected labels from ConfigMaps, filtered by asset type from the test's parametrize."""
    asset_type = request.param
    all_labels = get_labels_from_configmaps(admin_client=admin_client, namespace=model_registry_namespace)
    return [label for label in all_labels if label.get("assetType") == asset_type]


@pytest.fixture(scope="class")
def models_with_benchmark_data(
    model_catalog_pod: Pod,
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Return a dict of {model_name: model_data} for validated models that have benchmark metadata on the pod."""
    providers = model_catalog_pod.execute(
        command=["ls", PERFORMANCE_DATA_DIR],
        container=CATALOG_CONTAINER,
    )
    models_with_metadata: set[str] = set()
    for provider in providers.strip().splitlines():
        provider = provider.strip()
        if not provider or provider.endswith((".json", ".ndjson")):
            continue
        models_output = model_catalog_pod.execute(
            command=["ls", f"{PERFORMANCE_DATA_DIR}/{provider}"],
            container=CATALOG_CONTAINER,
        )
        for model in models_output.strip().splitlines():
            model = model.strip()
            if model and not model.endswith(".json"):
                models_with_metadata.add(f"{provider}/{model}")
    LOGGER.info(f"Found {len(models_with_metadata)} models with benchmark metadata on pod")

    api_response = execute_get_command(
        url=f"{model_catalog_rest_url[0]}models?source={VALIDATED_CATALOG_ID}&pageSize=100",
        headers=model_registry_rest_headers,
    )
    api_models = {model["name"]: model for model in api_response.get("items", [])}

    eligible = {name: api_models[name] for name in api_models if name in models_with_metadata}
    assert eligible, "No validated catalog models have benchmark metadata on the pod"
    LOGGER.info(f"Found {len(eligible)} eligible models with benchmark data")
    return eligible


# TODO: RHOAIENG-62057 - Remove this fixture and use randomly_picked_model_from_catalog_api_by_source
# once the bug is fixed.
@pytest.fixture(scope="class")
def model_with_benchmark_metadata(
    models_with_benchmark_data: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], str, str]:
    """Pick a random model from the validated catalog that has benchmark metadata on the pod."""
    model_name = random.choice(seq=list(models_with_benchmark_data.keys()))
    LOGGER.info(f"Picked model with benchmark metadata: {model_name}")
    return models_with_benchmark_data[model_name], model_name, VALIDATED_CATALOG_ID


@pytest.fixture(scope="class")
def validated_model_artifact_uris(
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
) -> dict[str, list[str]]:
    """Map of model name to its model-artifact URIs for all validated models."""
    models_response = execute_get_command(
        url=f"{model_catalog_rest_url[0]}models?source={VALIDATED_CATALOG_ID}&pageSize=1000",
        headers=model_registry_rest_headers,
    )
    models = models_response.get("items", [])
    assert models, f"No models found in {VALIDATED_CATALOG_ID} catalog"
    LOGGER.info(f"Fetching model-artifact URIs for {len(models)} validated models")

    result: dict[str, list[str]] = {}
    for model in models:
        model_name = model["name"]
        artifacts_url = (
            f"{model_catalog_rest_url[0]}sources/{VALIDATED_CATALOG_ID}"
            f"/models/{model_name}/artifacts?pageSize=100&artifactType={MODEL_ARTIFACT_TYPE}"
        )
        artifacts_response = execute_get_command(url=artifacts_url, headers=model_registry_rest_headers)
        result[model_name] = [artifact.get("uri", "") for artifact in artifacts_response.get("items", [])]

    return result


@pytest.fixture()
def validated_model(
    request: pytest.FixtureRequest,
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
) -> dict[str, Any]:
    """Fetch a validated model from the catalog API by name."""
    model_name = request.param
    response = get_models_from_catalog_api(
        model_catalog_rest_url=model_catalog_rest_url,
        model_registry_rest_headers=model_registry_rest_headers,
        page_size=1,
        additional_params=f"&source={VALIDATED_CATALOG_ID}&filterQuery=name='RedHatAI/{model_name}'",
    )
    items = response.get("items", [])
    assert items, f"Model '{model_name}' not found in catalog API for source '{VALIDATED_CATALOG_ID}'"
    return items[0]


@pytest.fixture()
def performance_artifacts(
    validated_model: dict[str, Any],
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
) -> list[dict[str, Any]]:
    """Fetch all performance artifacts for a validated model."""
    model_name = validated_model["name"]
    artifact_url = (
        f"{model_catalog_rest_url[0]}sources/{VALIDATED_CATALOG_ID}/models/{model_name}/artifacts/performance?pageSize"
    )
    response = fetch_all_artifacts_with_dynamic_paging(
        url_with_pagesize=artifact_url,
        headers=model_registry_rest_headers,
    )
    artifacts = response.get("items", [])
    assert artifacts, f"No performance artifacts found for '{model_name}'"
    return artifacts


@pytest.fixture(scope="class")
def model_detail_with_artifacts(
    request: pytest.FixtureRequest,
    models_with_benchmark_data: dict[str, dict[str, Any]],
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
) -> dict[str, Any]:
    """Fetch a model detail whose artifactCounts contains the required categories.

    Accepts parametrize with a set of required category keys, e.g.:
        {"model-artifact", "performance-metrics", "accuracy-metrics"}

    For categories that only need model-artifact (e.g. "other" models without benchmarks),
    searches all sources. For categories requiring benchmarks, searches models_with_benchmark_data.
    """
    required_categories: set[str] = getattr(request, "param", ALL_ARTIFACT_CATEGORIES)
    needs_benchmarks = required_categories - {"model-artifact"}

    if needs_benchmarks:
        candidates = [(VALIDATED_CATALOG_ID, name) for name in models_with_benchmark_data]
    else:
        # Try validated models first (already fetched), then other sources
        candidates = [(VALIDATED_CATALOG_ID, name) for name in models_with_benchmark_data]
        all_sources = execute_get_command(
            url=f"{model_catalog_rest_url[0]}sources",
            headers=model_registry_rest_headers,
        )
        for catalog_source in all_sources.get("items", []):
            source_id = catalog_source["id"]
            if source_id == VALIDATED_CATALOG_ID:
                continue
            models_response = execute_get_command(
                url=f"{model_catalog_rest_url[0]}models?source={source_id}&pageSize=50",
                headers=model_registry_rest_headers,
            )
            for model in models_response.get("items", []):
                candidates.append((source_id, model["name"]))

    for source_id, model_name in candidates:
        detail = execute_get_command(
            url=f"{model_catalog_rest_url[0]}sources/{source_id}/models/{model_name}",
            headers=model_registry_rest_headers,
        )
        artifact_counts = detail.get("artifactCounts", {})
        if set(artifact_counts.keys()) == required_categories:
            LOGGER.info(f"Selected model: {model_name} (source={source_id}), artifactCounts={artifact_counts}")
            return detail
        LOGGER.info(f"Skipping {model_name}: artifactCounts={artifact_counts}")

    pytest.fail(f"No model found with required artifact categories: {required_categories}")


@pytest.fixture(scope="class")
def expected_missing_categories(request: pytest.FixtureRequest) -> set[str]:
    """Return the set of artifact categories expected to be absent, derived from the required categories."""
    required = getattr(request, "param", ALL_ARTIFACT_CATEGORIES)
    return ALL_ARTIFACT_CATEGORIES - required
