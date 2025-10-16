from typing import Any

CUSTOM_CATALOG_ID1: str = "sample_custom_catalog1"
CUSTOM_CATALOG_ID2: str = "sample_custom_catalog2"
SAMPLE_MODEL_NAME1 = "mistralai/Mistral-7B-Instruct-v0.3"

SAMPLE_MODEL_NAME2 = "mistralai/Devstral-Small-2505"
EXPECTED_CUSTOM_CATALOG_VALUES: list[dict[str, str]] = [{"id": CUSTOM_CATALOG_ID1, "model_name": SAMPLE_MODEL_NAME1}]
MULTIPLE_CUSTOM_CATALOG_VALUES: list[dict[str, str]] = [
    {"id": CUSTOM_CATALOG_ID1, "model_name": SAMPLE_MODEL_NAME1},
    {"id": CUSTOM_CATALOG_ID2, "model_name": SAMPLE_MODEL_NAME2},
]

SAMPLE_MODEL_NAME3 = "mistralai/Ministral-8B-Instruct-2410"
CATALOG_CONTAINER: str = "catalog"
DEFAULT_CATALOGS: dict[str, Any] = {
    "redhat_ai_models": {
        "name": "Red Hat AI models",
        "type": "yaml",
        "properties": {"yamlCatalogPath": "/shared-data/models-catalog.yaml"},
    },
    "redhat_ai_validated_models": {
        "name": "Red Hat AI validated models",
        "type": "yaml",
        "properties": {"yamlCatalogPath": "/shared-data/validated-models-catalog.yaml"},
    },
}
REDHAT_AI_CATALOG_ID: str = next(iter(DEFAULT_CATALOGS))
DEFAULT_CATALOG_FILE: str = DEFAULT_CATALOGS[REDHAT_AI_CATALOG_ID]["properties"]["yamlCatalogPath"]
