from typing import Any

from kubernetes.dynamic import DynamicClient
from simple_logger.logger import get_logger

from tests.model_registry.model_catalog.db_constants import (
    GET_MODELS_BY_ACCURACY_DB_QUERY,
    GET_MODELS_BY_ACCURACY_WITH_TASK_FILTER_DB_QUERY,
)
from tests.model_registry.model_catalog.utils import (
    execute_database_query,
    get_models_from_catalog_api,
    parse_psql_output,
)
from tests.model_registry.utils import execute_get_command

LOGGER = get_logger(name=__name__)


def get_sources_with_sorting(
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
    order_by: str,
    sort_order: str,
) -> dict[str, Any]:
    """
    Get sources with sorting parameters

    Args:
        model_catalog_rest_url: REST URL for model catalog
        model_registry_rest_headers: Headers for model registry REST API
        order_by: Field to order results by (ID, NAME)
        sort_order: Sort order (ASC or DESC)

    Returns:
        Dictionary containing the API response
    """
    base_url = f"{model_catalog_rest_url[0]}sources"
    params = {
        "orderBy": order_by,
        "sortOrder": sort_order,
        "pageSize": 100,
    }

    return execute_get_command(url=base_url, headers=model_registry_rest_headers, params=params)


def get_artifacts_with_sorting(
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
    source_id: str,
    model_name: str,
    order_by: str,
    sort_order: str,
) -> dict[str, Any]:
    """
    Get artifacts with sorting parameters

    Args:
        model_catalog_rest_url: REST URL for model catalog
        model_registry_rest_headers: Headers for model registry REST API
        source_id: Source ID for the model
        model_name: Name of the model
        order_by: Field to order results by
        sort_order: Sort order (ASC or DESC)

    Returns:
        Dictionary containing the API response
    """
    base_url = f"{model_catalog_rest_url[0]}sources/{source_id}/models/{model_name}/artifacts"
    params = {
        "orderBy": order_by,
        "sortOrder": sort_order,
        "pageSize": 100,
    }

    return execute_get_command(url=base_url, headers=model_registry_rest_headers, params=params)


def validate_items_sorted_correctly(items: list[dict], field: str, order: str) -> bool:
    """
    Extract field values and verify they're in correct order

    Args:
        items: List of items to validate
        field: Field name to check sorting on
        order: Sort order (ASC or DESC)

    Returns:
        True if items are sorted correctly, False otherwise
    """

    # Extract field values for comparison
    values = []
    for item in items:
        if field == "ID":
            value = item.get("id")
        elif field == "NAME":
            value = item.get("name")
        elif field == "CREATE_TIME":
            value = item.get("createTimeSinceEpoch")
        elif field == "LAST_UPDATE_TIME":
            value = item.get("lastUpdateTimeSinceEpoch")
        else:
            raise ValueError(f"Invalid field: {field}")

        if value is None:
            raise ValueError(f"Field {field} is missing from item: {item}")

        values.append(value)
    # Convert values to appropriate types for comparison
    if field == "ID":
        # Convert IDs to integers for numeric comparison
        try:
            values = [int(value) for value in values]
        except ValueError:
            # If conversion fails, fall back to string comparison
            values = [str(value) for value in values]
    elif field == "NAME":
        # For NAME field, convert to lowercase for case-insensitive comparison
        values = [str(value).lower() for value in values]

    # Check if values are in correct order
    if order == "ASC":
        return all(values[i] <= values[i + 1] for i in range(len(values) - 1))
    elif order == "DESC":
        return all(values[i] >= values[i + 1] for i in range(len(values) - 1))
    else:
        raise ValueError(f"Invalid sort order: {order}")


def verify_custom_properties_sorted(items: list[dict], property_field: str, sort_order: str) -> bool:
    """
    Verify if a list of items is sorted by a specific custom property value.

    Expected sorting behavior:
    1. Items WITH the custom property appear first, sorted by property value (ASC or DESC)
    2. Items WITHOUT the custom property appear after, sorted by ID in ASC order

    Args:
        items: List of artifact items (with 'id' and 'customProperties')
        property_field: Property field to sort by (e.g., "e2e_p90.double_value")
        sort_order: "ASC" or "DESC" (applies only to items with the property)

    Returns:
        True if sorted correctly, False otherwise
    """
    property_name, value_type = property_field.rsplit(".", 1)
    # Separate items into two groups
    items_with_property, items_without_property = _split_items_by_custom_property(
        items=items, property_name=property_name, value_type=value_type, property_field=property_field
    )

    LOGGER.info(
        f"Total items: {len(items)}, with '{property_name}': {len(items_with_property)}, "
        f"without: {len(items_without_property)}"
    )

    # Verify items with property come first
    if items_with_property and items_without_property and items_with_property[-1][0] > items_without_property[0][0]:
        LOGGER.error(
            f"Items without property '{property_name}' appear before items with property. "
            f"Last with property at index {items_with_property[-1][0]}, "
            f"first without at index {items_without_property[0][0]}"
        )
        return False

    if items_with_property and not _verify_items_with_property_sorted(
        items=items_with_property, property_name=property_name, value_type=value_type, sort_order=sort_order
    ):
        return False

    if items_without_property and not _verify_items_without_property_sorted(items=items_without_property):
        return False

    LOGGER.info(f"All items sorted correctly by '{property_name}' ({sort_order})")
    return True


def validate_accuracy_sorting_against_database(
    admin_client: DynamicClient,
    api_response: dict[str, Any],
    sort_order: str | None,
    namespace: str = "rhoai-model-registries",
    task_filter: str | None = None,
) -> bool:
    """
    Validate API accuracy sorting results against database query results.

    Expected sorting behavior:
    - When sort_order is None:
      1. Models WITH accuracy appear first (in any order)
      2. Models WITHOUT accuracy appear after, sorted by model ID in ASC order
    - When sort_order is ASC or DESC:
      1. Models WITH accuracy appear first, sorted by accuracy value (ASC/DESC)
      2. Models WITHOUT accuracy appear after, sorted by model ID in ASC order

    Args:
        admin_client: DynamicClient with admin credentials
        api_response: API response from models endpoint with accuracy sorting
        sort_order: Sort order used (ASC, DESC, or None for no specific order)
        namespace: OpenShift namespace for PostgreSQL pod
        task_filter: Optional task filter value (e.g., "automatic-speech-recognition")

    Returns:
        True if sorted correctly, False otherwise
    """
    # Get models with accuracy from database
    models_with_accuracy = get_models_by_accuracy_from_database(
        admin_client=admin_client, sort_order=sort_order or "ASC", namespace=namespace, task_filter=task_filter
    )
    filter_info = f" with task filter '{task_filter}'" if task_filter else ""
    sort_info = f", ordered {sort_order}" if sort_order else " (no sort order)"
    LOGGER.info(f"Database query found {len(models_with_accuracy)} models with accuracy{filter_info}{sort_info}")

    # Get all models from API response (preserving order) - extract only name and id
    api_models = [(model.get("name"), model.get("id")) for model in api_response.get("items", [])]
    LOGGER.info(f"API returned {len(api_models)} total models")

    # Split API models into two groups: with accuracy and without accuracy
    models_with_accuracy_set = set(models_with_accuracy)
    api_models_with_accuracy, api_models_without_accuracy = _split_models_by_accuracy(
        api_models=api_models, models_with_accuracy_set=models_with_accuracy_set
    )

    LOGGER.info(
        f"API models split: {len(api_models_with_accuracy)} with accuracy, "
        f"{len(api_models_without_accuracy)} without accuracy"
    )

    # Validate: models with accuracy should come first
    if (
        api_models_with_accuracy
        and api_models_without_accuracy
        and api_models_with_accuracy[-1][0] > api_models_without_accuracy[0][0]
    ):
        LOGGER.error(
            f"Models without accuracy appear before models with accuracy. "
            f"Last with accuracy at index {api_models_with_accuracy[-1][0]}, "
            f"first without at index {api_models_without_accuracy[0][0]}"
        )
        return False

    # Validate: models with accuracy are in correct order (or present if no sort_order)
    if api_models_with_accuracy and not _verify_models_with_accuracy_sorted(
        models=api_models_with_accuracy, expected_models=models_with_accuracy, sort_order=sort_order
    ):
        return False

    # Validate: models without accuracy are sorted by ID ASC
    if api_models_without_accuracy and not _verify_items_without_property_sorted(api_models_without_accuracy):
        return False

    LOGGER.info(f"All models validated successfully{filter_info}{sort_info}")
    return True


def get_models_by_accuracy_from_database(
    admin_client: DynamicClient,
    sort_order: str,
    namespace: str = "rhoai-model-registries",
    task_filter: str | None = None,
) -> list[str]:
    """
    Query the database to get model names ordered by accuracy (overall_average).

    Args:
        admin_client: Admin client to use
        sort_order: Sort order for accuracy values (ASC or DESC)
        namespace: OpenShift namespace containing the PostgreSQL pod
        task_filter: Optional task filter value (e.g., "automatic-speech-recognition")

    Returns:
        List of model names ordered by accuracy
    """
    if task_filter:
        accuracy_query = GET_MODELS_BY_ACCURACY_WITH_TASK_FILTER_DB_QUERY.format(
            sort_order=sort_order, task_value=task_filter
        )
    else:
        accuracy_query = GET_MODELS_BY_ACCURACY_DB_QUERY.format(sort_order=sort_order)

    LOGGER.debug(f"Accuracy query (SQL): {accuracy_query}")

    # Execute the database query
    db_result = execute_database_query(admin_client=admin_client, query=accuracy_query, namespace=namespace)
    parsed_result = parse_psql_output(psql_output=db_result)

    # The query returns context_name values in order
    return parsed_result.get("values", [])


def _split_items_by_custom_property(
    items: list[dict],
    property_name: str,
    value_type: str,
    property_field: str,
) -> tuple[list[dict], list[dict]]:
    """
    Split items into two lists based on the presence of a custom property.
    """
    items_with_property = []
    items_without_property = []

    for idx, item in enumerate(items):
        custom_props = item.get("customProperties", {})
        if property_name in custom_props:
            prop = custom_props[property_name]
            value = prop.get(value_type) if isinstance(prop, dict) else None

            # Only add to items_with_property if value is not None
            if value is not None:
                LOGGER.info(f"  [{idx}] ID: {item['id']}, {property_field}: {value}")
                items_with_property.append((idx, item))
            else:
                LOGGER.info(f"  [{idx}] ID: {item['id']} (has {property_name} but value is None)")
                items_without_property.append((idx, item))
        else:
            LOGGER.info(f"  [{idx}] ID: {item['id']} (no {property_name})")
            items_without_property.append((idx, item))
    return items_with_property, items_without_property


def _verify_items_with_property_sorted(items: list[dict], property_name: str, value_type: str, sort_order: str) -> bool:
    """
    Verify if items with property are sorted correctly by property value
    """
    values = []
    for _, item in items:
        prop = item["customProperties"][property_name]
        value = prop.get(value_type)
        values.append(value)

    expected_values = sorted(values, reverse=(sort_order == "DESC"))
    if values != expected_values:
        LOGGER.error(f"Items with property not sorted {sort_order}: {values}")
        return False
    return True


def _verify_items_without_property_sorted(items: list[dict]) -> bool:
    """
    Verify if items without property are sorted correctly by ID in ASC order
    """
    ids = [int(item["id"]) for _, item in items]
    expected_ids = sorted(ids)
    if ids != expected_ids:
        LOGGER.error(f"Items without property not sorted by ID ASC: {ids}")
        return False
    return True


def _split_models_by_accuracy(
    api_models: list[tuple[str, str]], models_with_accuracy_set: set[str]
) -> tuple[list[tuple[int, str]], list[tuple[int, dict[str, str]]]]:
    """
    Split API models into two groups based on accuracy property presence.

    Args:
        api_models: List of (name, id) tuples from API response
        models_with_accuracy_set: Set of model names that have accuracy

    Returns:
        Tuple of (models_with_accuracy, models_without_accuracy)
        - models_with_accuracy: list of (index, name) tuples
        - models_without_accuracy: list of (index, {"id": model_id}) tuples
    """
    models_with_accuracy = []
    models_without_accuracy = []

    for idx, (name, model_id) in enumerate(api_models):
        if name in models_with_accuracy_set:
            models_with_accuracy.append((idx, name))
        else:
            models_without_accuracy.append((idx, {"id": model_id}))

    return models_with_accuracy, models_without_accuracy


def _verify_models_with_accuracy_sorted(
    models: list[tuple[int, str]], expected_models: list[str], sort_order: str | None
) -> bool:
    """
    Verify if models with accuracy are sorted correctly by accuracy value.
    When sort_order is None, only verifies that the expected models are present (order not validated).

    Args:
        models: List of (index, name) tuples for models with accuracy
        expected_models: Expected list of model names from database
        sort_order: Sort order (ASC, DESC, or None for no order validation)

    Returns:
        True if sorted correctly, False otherwise
    """
    if sort_order:
        # Validate specific order
        actual_names = [name for _, name in models]
        if actual_names != expected_models:
            LOGGER.error(
                f"Models with accuracy in wrong order ({sort_order}):\n"
                f"  Expected: {expected_models}\n"
                f"  Actual:   {actual_names}"
            )
            return False
    else:
        # Only validate presence, not order
        actual_names = {name for _, name in models}
        expected_names = set(expected_models)
        if actual_names != expected_names:
            LOGGER.error("Models with accuracy do not match expected models from database")
            return False
    return True


def get_minimum_artifact_property_value(
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
    source_id: str,
    model_name: str,
    property_field: str,
    artifact_filter_query: str | None = None,
    sort_order: str = "ASC",
    recommendations: bool = False,
) -> float | None:
    """
    Get the minimum value of an artifact property for a given model.

    Args:
        model_catalog_rest_url: REST URL for model catalog
        model_registry_rest_headers: Headers for model registry REST API
        source_id: Source ID for the model
        model_name: Name of the model
        property_field: Property field to get minimum value for
        artifact_filter_query: Optional filter query for artifacts (without "artifacts." prefix)
        sort_order: Sort order for the artifact property
        recommendations: Whether to filter to only recommended artifacts

    Returns:
        Minimum property value, or None if no artifacts found
    """
    # Build the artifacts endpoint URL
    base_url = f"{model_catalog_rest_url[0]}sources/{source_id}/models/{model_name}/artifacts/performance"

    params = {
        "orderBy": property_field,
        "sortOrder": sort_order,
        "pageSize": 1,
    }

    if artifact_filter_query:
        params["filterQuery"] = artifact_filter_query

    if recommendations:
        params["recommendations"] = "true"

    response = execute_get_command(url=base_url, headers=model_registry_rest_headers, params=params)

    items = response.get("items", [])
    if not items:
        return None

    # Extract the property value from the first (minimum) artifact
    property_name, value_type = property_field.rsplit(".", 1)
    custom_props = items[0].get("customProperties", {})

    if property_name not in custom_props:
        return None

    prop = custom_props[property_name]
    return prop.get(value_type) if isinstance(prop, dict) else None


def get_model_latencies(
    model_names: list[str],
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
    source_id: str,
    property_field: str,
    artifact_filter_query: str,
    sort_order: str,
    recommendations: bool,
) -> list[float]:
    """
    Fetch minimum artifact property values for a list of models.

    Args:
        model_names: List of model names
        model_catalog_rest_url: REST URL for model catalog
        model_registry_rest_headers: Headers for model registry REST API
        source_id: Source ID for the models
        property_field: Property field to get minimum value for
        artifact_filter_query: Filter query for artifacts (without "artifacts." prefix)
        sort_order: Sort order for the artifact property
        recommendations: Whether to filter to only recommended artifacts

    Returns:
        List of minimum property values for each model
    """
    latencies = []
    for model_name in model_names:
        latency = get_minimum_artifact_property_value(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            source_id=source_id,
            model_name=model_name,
            property_field=property_field,
            artifact_filter_query=artifact_filter_query,
            sort_order=sort_order,
            recommendations=recommendations,
        )
        latencies.append(latency)
    return latencies


def assert_model_sorting(
    order_by: str,
    sort_order: str | None,
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
) -> None:
    LOGGER.info(f"Testing models sorting: orderBy={order_by}, sortOrder={sort_order}")

    response = get_models_from_catalog_api(
        model_catalog_rest_url=model_catalog_rest_url,
        model_registry_rest_headers=model_registry_rest_headers,
        order_by=order_by,
        sort_order=sort_order,
    )

    assert validate_items_sorted_correctly(items=response["items"], field=order_by, order=sort_order)
