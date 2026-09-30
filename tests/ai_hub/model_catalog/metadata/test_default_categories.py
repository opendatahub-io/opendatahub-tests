from typing import Any

import pytest
import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap

from tests.ai_hub.constants import DEFAULT_MODEL_CATALOG_CM
from tests.ai_hub.model_catalog.constants import (
    DEFAULT_CATALOGS,
    OTHER_MODELS_CATALOG_NAME,
    RETIRED_CATALOG_ID,
    VALIDATED_CATALOG_LABEL,
    VALIDATED_CATALOG_NAME,
)
from tests.ai_hub.model_catalog.utils import get_all_catalog_items

pytestmark = [
    pytest.mark.tier1,
    pytest.mark.install,
    pytest.mark.post_upgrade,
    pytest.mark.usefixtures("updated_dsc_component_state_scope_session", "model_registry_namespace"),
]


def test_default_category_configuration(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
) -> None:
    """Given the shipped category configuration,
    When reading its ConfigMap and API sources,
    Then only the two supported default categories are configured with the expected names and labels.
    """
    configmap = ConfigMap(name=DEFAULT_MODEL_CATALOG_CM, namespace=model_registry_namespace, client=admin_client)
    configured = yaml.safe_load(configmap.instance.data["sources.yaml"])["catalogs"]
    assert len(configured) == len(DEFAULT_CATALOGS)
    assert {source["id"] for source in configured} == set(DEFAULT_CATALOGS)
    sources = get_all_catalog_items(url=f"{model_catalog_rest_url[0]}sources", headers=model_registry_rest_headers)
    api_sources = {source["id"]: source for source in sources}
    assert RETIRED_CATALOG_ID not in api_sources, "Retired default source is still exposed"
    for source in configured:
        source_id = source["id"]
        assert source_id in api_sources, f"Default source {source_id} is missing from the API"
        expected = DEFAULT_CATALOGS[source_id]
        for field in ("name", "labels", "enabled"):
            assert source.get(field) == expected[field], f"Incorrect configured {field} for {source_id}"
            api_expected = (expected[field] or []) if field == "labels" else expected[field]
            assert api_sources[source_id].get(field) == api_expected, f"Incorrect API {field} for {source_id}"
        assert api_sources[source_id]["status"] == "available"


def test_default_category_display_names(
    model_catalog_rest_url: list[str], model_registry_rest_headers: dict[str, str]
) -> None:
    """Given the default model categories,
    When reading model labels,
    Then their stable label keys have the expected display names and the retired category is absent.
    """
    labels = get_all_catalog_items(
        url=f"{model_catalog_rest_url[0]}labels", headers=model_registry_rest_headers, params={"assetType": "models"}
    )
    names = {label["name"]: label.get("displayName") for label in labels}
    assert names[VALIDATED_CATALOG_LABEL] == VALIDATED_CATALOG_NAME
    assert names[None] == OTHER_MODELS_CATALOG_NAME
    assert "Red Hat AI" not in names, "Retired model category label is still exposed"


def test_default_catalog_model_membership(
    shipped_default_catalog_models: dict[str, list[dict[str, Any]]],
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
) -> None:
    """Given both shipped catalogs,
    When reading every model page for each source,
    Then the API exposes exactly their models without duplicates within or across default sources.
    """
    seen_names: set[str] = set()
    for source_id, shipped_models in shipped_default_catalog_models.items():
        expected_names = {model["name"] for model in shipped_models}
        assert expected_names, f"Shipped catalog {source_id} is empty"
        assert len(expected_names) == len(shipped_models), f"Duplicate names in shipped catalog {source_id}"
        assert not seen_names & expected_names, (
            f"Duplicate models across default catalogs: {seen_names & expected_names}"
        )
        seen_names.update(expected_names)
        models = get_all_catalog_items(
            url=f"{model_catalog_rest_url[0]}models", headers=model_registry_rest_headers, params={"source": source_id}
        )
        actual_names = {model["name"] for model in models}
        assert actual_names == expected_names, (
            f"Model membership differs for {source_id}: {actual_names ^ expected_names}"
        )
        assert len(models) == len(expected_names), f"Duplicate API models for {source_id}"
        assert all(model["source_id"] == source_id for model in models)
