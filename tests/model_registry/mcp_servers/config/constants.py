DEFAULT_MCP_LABEL: str = "Red Hat"

EXPECTED_DEFAULT_MCP_CATALOG: dict = {
    "name": "Red Hat MCP Servers",
    "id": "rh_mcp_servers",
    "type": "yaml",
    "enabled": True,
    "properties": {
        "yamlCatalogPath": "/shared-data/redhat-mcp-servers-catalog.yaml",
    },
    "labels": ["Red Hat"],
}
