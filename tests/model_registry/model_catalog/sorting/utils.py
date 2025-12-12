from typing import Any

from simple_logger.logger import get_logger
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
            values = [int(v) for v in values]
        except ValueError:
            # If conversion fails, fall back to string comparison
            values = [str(v) for v in values]

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
