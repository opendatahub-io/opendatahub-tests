from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient

from tests.model_registry.model_catalog.metadata.utils import get_labels_from_configmaps


@pytest.fixture()
def expected_labels_by_asset_type(
    admin_client: DynamicClient,
    model_registry_namespace: str,
) -> dict[str, list[dict[str, Any]]]:
    """Get expected labels from ConfigMaps, split by asset type."""
    all_labels = get_labels_from_configmaps(admin_client=admin_client, namespace=model_registry_namespace)
    mcp_labels = [label for label in all_labels if label.get("assetType") == "mcp_servers"]
    model_labels = [label for label in all_labels if label not in mcp_labels]
    return {"models": model_labels, "mcp_servers": mcp_labels}
