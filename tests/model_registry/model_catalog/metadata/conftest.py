import random
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.pod import Pod

from tests.model_registry.model_catalog.constants import (
    CATALOG_CONTAINER,
    PERFORMANCE_DATA_DIR,
    VALIDATED_CATALOG_ID,
)
from tests.model_registry.model_catalog.metadata.utils import get_labels_from_configmaps
from tests.model_registry.utils import execute_get_command

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
