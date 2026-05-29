import pytest


@pytest.fixture(scope="class")
def catalog_base_url(model_catalog_rest_url: list[str]) -> str:
    """Base URL for the catalog server without the API path."""
    return model_catalog_rest_url[0].split("/api/")[0]
