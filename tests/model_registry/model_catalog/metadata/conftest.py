from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient

from tests.model_registry.model_catalog.metadata.utils import get_labels_from_configmaps


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
