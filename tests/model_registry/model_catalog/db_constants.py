# Constants useful for querying the model catalog database and parsing its responses

# SQL query for filter_options endpoint database validation
# Queries materialized views (context_property_options, artifact_property_options) that aggregate
# filterable properties for CatalogModel. Based on GetFilterableProperties in
# kubeflow/model-registry catalog/internal/db/service/catalog_model.go (PR #1875)
#
# Property naming:
# - Context properties: base name only (special case: 'validated_on.array_value' for arrays)
# - Artifact properties: 'artifacts.{name}.{type}' where type is string_value/array_value/double_value/int_value
#
# Return format:
# - String/array properties: text array of values
# - Numeric properties: 2-element text array [min, max] converted from double/int columns
FILTER_OPTIONS_DB_QUERY = """
SELECT
    CASE
        WHEN name = 'validated_on' AND array_value IS NOT NULL THEN name || '.array_value'
        ELSE name
    END AS name,
    COALESCE(string_value, array_value, '{}'::text[]) AS array_agg
FROM context_property_options
WHERE type_id = (SELECT id FROM "Type" WHERE name = 'kf.CatalogModel')

UNION ALL

SELECT
    'artifacts.' ||
    CASE
        WHEN string_value IS NOT NULL THEN name || '.string_value'
        WHEN array_value IS NOT NULL THEN name || '.array_value'
        WHEN min_double_value IS NOT NULL THEN name || '.double_value'
        WHEN min_int_value IS NOT NULL THEN name || '.int_value'
        ELSE name
    END AS name,
    CASE
        WHEN min_double_value IS NOT NULL THEN
            ARRAY[min_double_value::text, max_double_value::text]
        WHEN min_int_value IS NOT NULL THEN
            ARRAY[min_int_value::text, max_int_value::text]
        ELSE
            COALESCE(string_value, array_value, '{}'::text[])
    END AS array_agg
FROM artifact_property_options

ORDER BY name;
"""

# SQL query for search functionality database validation
# Replicates the exact database query used by applyCatalogModelListFilters for the search/q parameter
# in kubeflow/model-registry catalog/internal/db/service/catalog_model.go
# Note: Uses parameterized pattern that should be formatted with the search pattern
SEARCH_MODELS_DB_QUERY = """
SELECT DISTINCT c.id as model_id
FROM "Context" c
WHERE c.type_id = (SELECT id FROM "Type" WHERE name = 'kf.CatalogModel')
AND (
    LOWER(c.name) LIKE '{search_pattern}'
    OR
    EXISTS (
        SELECT 1
        FROM "ContextProperty" cp
        WHERE cp.context_id = c.id
        AND cp.name IN ('description', 'provider', 'libraryName')
        AND LOWER(cp.string_value) LIKE '{search_pattern}'
    )
    OR
    EXISTS (
        SELECT 1
        FROM "ContextProperty" cp
        WHERE cp.context_id = c.id
        AND cp.name = 'tasks'
        AND LOWER(cp.string_value) LIKE '{search_pattern}'
    )
)
ORDER BY model_id;
"""

# SQL query for search functionality with source_id filtering database validation
# Extends SEARCH_MODELS_DB_QUERY to include source_id filtering for specific catalog sources
# Note: Uses parameterized patterns for both search_pattern and source_ids (comma-separated quoted list)
SEARCH_MODELS_WITH_SOURCE_ID_DB_QUERY = """
SELECT DISTINCT c.id as model_id
FROM "Context" c
WHERE c.type_id = (SELECT id FROM "Type" WHERE name = 'kf.CatalogModel')
AND (
    LOWER(c.name) LIKE '{search_pattern}'
    OR
    EXISTS (
        SELECT 1
        FROM "ContextProperty" cp
        WHERE cp.context_id = c.id
        AND cp.name IN ('description', 'provider', 'libraryName')
        AND LOWER(cp.string_value) LIKE '{search_pattern}'
    )
    OR
    EXISTS (
        SELECT 1
        FROM "ContextProperty" cp
        WHERE cp.context_id = c.id
        AND cp.name = 'tasks'
        AND LOWER(cp.string_value) LIKE '{search_pattern}'
    )
)
AND EXISTS (
    SELECT 1
    FROM "ContextProperty" cp
    WHERE cp.context_id = c.id
    AND cp.name = 'source_id'
    AND cp.string_value IN ({source_ids})
)
ORDER BY model_id;
"""

# SQL query for filterQuery parameter database validation with license filter only
# Replicates the database query used by the filterQuery parameter functionality
# for the specific pattern: license IN (...)
# Note: Uses {licenses} placeholder
FILTER_MODELS_BY_LICENSE_DB_QUERY = """
SELECT DISTINCT c.id as model_id
FROM "Context" c
WHERE c.type_id = (SELECT id FROM "Type" WHERE name = 'kf.CatalogModel')
AND EXISTS (
    SELECT 1
    FROM "ContextProperty" cp
    WHERE cp.context_id = c.id
    AND cp.name = 'license'
    AND cp.string_value IN ({licenses})
)
ORDER BY model_id;
"""

# SQL query for filterQuery parameter database validation with license and language filters
# Replicates the database query used by the filterQuery parameter functionality
# for the specific pattern: license IN (...) AND (language ILIKE ... OR language ILIKE ...)
# Note: Uses {licenses}, {language_pattern_1}, {language_pattern_2} placeholders
FILTER_MODELS_BY_LICENSE_AND_LANGUAGE_DB_QUERY = """
SELECT DISTINCT c.id as model_id
FROM "Context" c
WHERE c.type_id = (SELECT id FROM "Type" WHERE name = 'kf.CatalogModel')
AND EXISTS (
    SELECT 1
    FROM "ContextProperty" cp
    WHERE cp.context_id = c.id
    AND cp.name = 'license'
    AND cp.string_value IN ({licenses})
)
AND (
    EXISTS (
        SELECT 1
        FROM "ContextProperty" cp
        WHERE cp.context_id = c.id
        AND cp.name = 'language'
        AND LOWER(cp.string_value) LIKE LOWER('{language_pattern_1}')
    )
    OR
    EXISTS (
        SELECT 1
        FROM "ContextProperty" cp
        WHERE cp.context_id = c.id
        AND cp.name = 'language'
        AND LOWER(cp.string_value) LIKE LOWER('{language_pattern_2}')
    )
)
ORDER BY model_id;
"""

# Fields that are explicitly filtered out by the filter_options endpoint API
# From db_catalog.go in kubeflow/model-registry GetFilterOptions method
# Updated with PR #1875 to include metricsType and model_id exclusions
API_EXCLUDED_FILTER_FIELDS = {
    "source_id",
    "logo",
    "license_link",
    "artifacts.metricsType.string_value",  # artifact property with full name
    "artifacts.model_id.string_value",  # artifact property with full name
}
