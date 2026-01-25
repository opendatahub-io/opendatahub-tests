# Constants useful for querying the model catalog database and parsing its responses

# SQL query for filter_options endpoint database validation
# Queries materialized views (context_property_options, artifact_property_options) that aggregate
# filterable properties for CatalogModel. Based on GetFilterableProperties in
# kubeflow/model-registry catalog/internal/db/service/catalog_model.go (PR #1875)
#
# Query structure:
# 1. First SELECT: Context properties (model-level metadata)
#    - Core properties: Return base name only (e.g., license, provider, tasks)
#    - Custom properties with known types: Add type suffix (e.g., model_type.string_value, validated_on.array_value)
# 2. UNION ALL
# 3. Second SELECT: Artifact properties (model artifact metadata)
#    - Format: 'artifacts.{property_name}.{value_type}'
#    - Value types: string_value, array_value, double_value, int_value
#
# Return format:
# - String/array properties: PostgreSQL text array of distinct values
# - Numeric properties: 2-element text array [min, max] converted from double/int columns
FILTER_OPTIONS_DB_QUERY = """
SELECT
    CASE
        -- Custom properties with array_value get .array_value suffix
        WHEN name IN ('validated_on') AND array_value IS NOT NULL THEN name || '.array_value'
        -- Custom properties with string_value get .string_value suffix
        WHEN name IN ('model_type', 'size', 'tensor_type', 'variant_group_id') THEN name || '.string_value'
        -- Core properties keep base name only
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

# Fields that are dynamically computed and added by the API but do not exist in the database
API_COMPUTED_FILTER_FIELDS = {
    "status",  # Computed from CatalogSource.status field
}

# SQL query for accuracy sorting database validation
# Returns an ordered list of model names (context names) that have accuracy metrics.
# Models are ordered by their overall_average (accuracy) value from artifact properties.
# Only returns the model name column for easy comparison with API results.
GET_MODELS_BY_ACCURACY_DB_QUERY = """
SELECT c.name
FROM "ArtifactProperty" ap
JOIN "Artifact" a ON a.id = ap.artifact_id
JOIN "Attribution" attr ON attr.artifact_id = a.id
JOIN "Context" c ON c.id = attr.context_id
WHERE ap.name ILIKE '%average%'
ORDER BY ap.double_value {sort_order};
"""

# SQL query for accuracy sorting with task filter database validation
# Returns an ordered list of model names (context names) that have accuracy metrics
# and match the specified task filter.
# Models are ordered by their overall_average (accuracy) value from artifact properties.
# The tasks field is stored as a JSON array, so we use LIKE pattern matching
GET_MODELS_BY_ACCURACY_WITH_TASK_FILTER_DB_QUERY = """
SELECT c.name
FROM "ArtifactProperty" ap
JOIN "Artifact" a ON a.id = ap.artifact_id
JOIN "Attribution" attr ON attr.artifact_id = a.id
JOIN "Context" c ON c.id = attr.context_id
WHERE ap.name ILIKE '%average%'
AND EXISTS (
    SELECT 1
    FROM "ContextProperty" cp
    WHERE cp.context_id = c.id
    AND cp.name = 'tasks'
    AND cp.string_value LIKE '%"{task_value}"%'
)
ORDER BY ap.double_value {sort_order};
"""

# SQL query for getting models by source ID
GET_MODELS_BY_SOURCE_ID_DB_QUERY = """
SELECT DISTINCT c.name as model_name
FROM "Context" c
WHERE c.type_id = (SELECT id FROM "Type" WHERE name = 'kf.CatalogModel')
AND EXISTS (
    SELECT 1
    FROM "ContextProperty" cp
    WHERE cp.context_id = c.id
    AND cp.name = 'source_id'
    AND cp.string_value = '{source_id}'
)
ORDER BY model_name;
"""
