from typing import Any

import structlog

from tests.ai_hub.mcp_servers.preview.constants import MCP_SERVERS_INLINE_DATA
from tests.ai_hub.utils import execute_authenticated_post, should_include_by_pattern

LOGGER = structlog.get_logger(name=__name__)


def post_stateless_preview(
    model_catalog_preview_url: str,
    token: str,
    page_size: int = 100,
    next_page_token: str = "",
    included_servers: list[str] | None = None,
    excluded_servers: list[str] | None = None,
) -> dict[str, Any]:
    """Post a stateless MCP preview request with inline catalog data."""
    config_content = build_mcp_preview_config(
        included_servers=included_servers,
        excluded_servers=excluded_servers,
    )

    query = f"pageSize={page_size}"
    if next_page_token:
        query += f"&nextPageToken={next_page_token}"
    url = f"{model_catalog_preview_url}sources/preview?{query}"

    files = {
        "config": ("config.yaml", config_content, "application/x-yaml"),
        "catalogData": ("catalog-data.yaml", MCP_SERVERS_INLINE_DATA, "application/x-yaml"),
    }
    return execute_authenticated_post(url=url, token=token, files=files)


def build_mcp_preview_config(
    yaml_catalog_path: str | None = None,
    included_servers: list[str] | None = None,
    excluded_servers: list[str] | None = None,
) -> str:
    """Build MCP preview config YAML content."""
    config_lines = ["type: yaml", "assetType: mcp_servers"]

    if yaml_catalog_path:
        config_lines.extend([
            "properties:",
            f"  yamlCatalogPath: {yaml_catalog_path}",
        ])

    if included_servers:
        config_lines.append("includedServers:")
        config_lines.extend(f'  - "{pattern}"' for pattern in included_servers)

    if excluded_servers:
        config_lines.append("excludedServers:")
        config_lines.extend(f'  - "{pattern}"' for pattern in excluded_servers)

    return "\n".join(config_lines)


def validate_mcp_preview_counts(
    result: dict[str, Any],
    mcp_servers: list[dict[str, Any]],
    included_patterns: list[str] | None = None,
    excluded_patterns: list[str] | None = None,
) -> None:
    """Validate MCP preview API summary counts against expected YAML content."""
    summary = result.get("summary")
    assert summary is not None, f"Missing 'summary' field in API response (keys: {list(result.keys())})"

    total = len(mcp_servers)
    included_count = sum(
        1
        for server in mcp_servers
        if should_include_by_pattern(
            name=server.get("name", ""), included_patterns=included_patterns, excluded_patterns=excluded_patterns
        )
    )
    expected = {"totalAssets": total, "includedAssets": included_count, "excludedAssets": total - included_count}
    LOGGER.info(f"MCP preview counts expected: {expected}, actual: {summary}")

    errors = [
        f"{key} mismatch: API={summary[key]}, expected={expected[key]}"
        for key in expected
        if summary[key] != expected[key]
    ]
    assert not errors, "Validation failures:\n" + "\n".join(f"  - {err}" for err in errors)
    LOGGER.info(f"MCP preview validation passed: {expected}")


def validate_preview_items(
    result: dict[str, Any],
    included_patterns: list[str] | None = None,
    excluded_patterns: list[str] | None = None,
    expected_server_names: set[str] | None = None,
    allow_empty: bool = False,
) -> None:
    """Validate that each item in the preview response has the correct 'included' property.

    Args:
        result: API response from preview endpoint
        included_patterns: Glob patterns for includedServers (None means include all)
        excluded_patterns: Glob patterns for excludedServers (None means exclude none)
        expected_server_names: If provided, verify all returned names exist in this set
        allow_empty: If True, skip the non-empty items assertion (e.g. for pagination last page)
    """
    assert result.get("assetType") == "mcp_servers", (
        f"Expected assetType 'mcp_servers', got '{result.get('assetType')}'"
    )
    items = result.get("items", [])
    if not allow_empty:
        assert items, "Preview response returned empty items list"
    LOGGER.info(f"Filters: includedServers={included_patterns}, excludedServers={excluded_patterns}")
    item_details = [{"name": item["name"], "included": item.get("included")} for item in items]
    LOGGER.info(f"Validating {len(items)} items: {item_details}")

    errors = []
    for item in items:
        name = item.get("name", "")
        if expected_server_names is not None and name not in expected_server_names:
            errors.append(f"Server '{name}': not found in catalog (known: {expected_server_names})")
        item_included = item.get("included")
        if item_included is None:
            errors.append(f"Server '{name}': missing 'included' property")
            continue
        expected_included = should_include_by_pattern(
            name=name, included_patterns=included_patterns, excluded_patterns=excluded_patterns
        )
        if item_included != expected_included:
            errors.append(f"Server '{name}': included={item_included}, expected={expected_included}")

    assert not errors, f"Found {len(errors)} items with incorrect 'included' property:\n" + "\n".join(errors)
    LOGGER.info(f"All {len(items)} items have correct 'included' property")
