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


# TODO: RHOAIENG-62057 - Remove this fixture and use randomly_picked_model_from_catalog_api_by_source
# once the bug is fixed.
@pytest.fixture(scope="class")
def model_with_benchmark_metadata(
    model_catalog_pod: Pod,
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
) -> tuple[dict[Any, Any], str, str]:
    """Pick a random model from the validated catalog that has benchmark metadata on the pod."""
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

    eligible = [name for name in api_models if name in models_with_metadata]
    assert eligible, "No validated catalog models have benchmark metadata on the pod"

    model_name = random.choice(seq=eligible)
    LOGGER.info(f"Picked model with benchmark metadata: {model_name}")
    return api_models[model_name], model_name, VALIDATED_CATALOG_ID


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
