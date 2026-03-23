from typing import Any

import pytest

from tests.model_registry.model_catalog.utils import get_models_from_catalog_api


@pytest.fixture(scope="class")
def embedding_models_response(
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
) -> dict[str, Any]:
    """Fetch models filtered by tasks='text-embedding' via filterQuery"""
    return get_models_from_catalog_api(
        model_catalog_rest_url=model_catalog_rest_url,
        model_registry_rest_headers=model_registry_rest_headers,
        additional_params="&filterQuery=tasks='text-embedding'",
    )
