# Constants useful for querying the model catalog database and parsing its responses

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
