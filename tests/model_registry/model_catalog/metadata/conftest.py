from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient

from tests.model_registry.model_catalog.metadata.utils import get_labels_from_configmaps
from tests.model_registry.utils import execute_get_command


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
def tool_calling_model_readme(
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
) -> tuple[str, str]:
    """
    Fetch tool-calling model README once per test class to avoid redundant API calls.

    Returns:
        Tuple of (readme_content, model_name)
    """
    model_name = "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic"
    catalog_id = "redhat_ai_validated_models"

    model_data = execute_get_command(
        url=f"{model_catalog_rest_url[0]}sources/{catalog_id}/models/{model_name}",
        headers=model_registry_rest_headers,
    )
    return model_data.get("readme", ""), model_name
