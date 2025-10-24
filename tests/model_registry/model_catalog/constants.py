from typing import Any

from tests.model_registry.constants import SAMPLE_MODEL_NAME1, CUSTOM_CATALOG_ID1

CUSTOM_CATALOG_ID2: str = "sample_custom_catalog2"

SAMPLE_MODEL_NAME2 = "mistralai/Devstral-Small-2505"
EXPECTED_CUSTOM_CATALOG_VALUES: list[dict[str, str]] = [{"id": CUSTOM_CATALOG_ID1, "model_name": SAMPLE_MODEL_NAME1}]
MULTIPLE_CUSTOM_CATALOG_VALUES: list[dict[str, str]] = [
    {"id": CUSTOM_CATALOG_ID1, "model_name": SAMPLE_MODEL_NAME1},
    {"id": CUSTOM_CATALOG_ID2, "model_name": SAMPLE_MODEL_NAME2},
]

REDHAT_AI_CATALOG_NAME: str = "Red Hat AI"
REDHAT_AI_VALIDATED_CATALOG_NAME: str = "Red Hat AI validated"

SAMPLE_MODEL_NAME3 = "mistralai/Ministral-8B-Instruct-2410"
CATALOG_CONTAINER: str = "catalog"
DEFAULT_CATALOGS: dict[str, Any] = {
    "redhat_ai_models": {
        "name": REDHAT_AI_CATALOG_NAME,
        "type": "yaml",
        "properties": {"yamlCatalogPath": "/shared-data/models-catalog.yaml"},
        "labels": [REDHAT_AI_CATALOG_NAME],
    },
    "redhat_ai_validated_models": {
        "name": REDHAT_AI_VALIDATED_CATALOG_NAME,
        "type": "yaml",
        "properties": {"yamlCatalogPath": "/shared-data/validated-models-catalog.yaml"},
        "labels": [REDHAT_AI_VALIDATED_CATALOG_NAME],
    },
}
REDHAT_AI_CATALOG_ID: str = "redhat_ai_models"
DEFAULT_CATALOG_FILE: str = DEFAULT_CATALOGS[REDHAT_AI_CATALOG_ID]["properties"]["yamlCatalogPath"]
VALIDATED_CATALOG_ID: str = "redhat_ai_validated_models"

REDHAT_AI_FILTER: str = "Red+Hat+AI"
REDHAT_AI_VALIDATED_FILTER = "Red+Hat+AI+Validated"

# SQL query for filter_options endpoint database validation
# Replicates the exact database query used by GetFilterableProperties for the filter_options endpoint
# in kubeflow/model-registry catalog/internal/db/service/catalog_model.go
# Note: Uses dynamic type_id lookup via 'kf.CatalogModel' name since type_id appears to be dynamic
FILTER_OPTIONS_DB_QUERY = """
SELECT name, array_agg(string_value) FROM (
    SELECT
        name,
        string_value
    FROM "ContextProperty" WHERE
        context_id IN (
            SELECT id FROM "Context" WHERE type_id = (
                SELECT id FROM "Type" WHERE name = 'kf.CatalogModel'
            )
        )
        AND string_value IS NOT NULL
        AND string_value != ''
        AND string_value IS NOT JSON ARRAY

    UNION

    SELECT
        name,
        json_array_elements_text(string_value::json) AS string_value
    FROM "ContextProperty" WHERE
        context_id IN (
            SELECT id FROM "Context" WHERE type_id = (
                SELECT id FROM "Type" WHERE name = 'kf.CatalogModel'
            )
        )
        AND string_value IS JSON ARRAY
)
GROUP BY name HAVING MAX(CHAR_LENGTH(string_value)) <= 100;
"""

# Fields that are explicitly filtered out by the filter_options endpoint API
# From db_catalog.go:204-206 in kubeflow/model-registry GetFilterOptions method
API_EXCLUDED_FILTER_FIELDS = {"source_id", "logo", "license_link"}
