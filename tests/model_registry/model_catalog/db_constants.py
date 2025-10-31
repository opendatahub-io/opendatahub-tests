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
# From db_catalog.go:204-206 in kubeflow/model-registry GetFilterOptions method
API_EXCLUDED_FILTER_FIELDS = {"source_id", "logo", "license_link"}
