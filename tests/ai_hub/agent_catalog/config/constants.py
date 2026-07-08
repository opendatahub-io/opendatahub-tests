AGENT_CATALOG_SOURCES_CM: str = "agent-catalog-sources"
AGENT_CATALOG_API_PATH: str = "/api/agent_catalog/v1alpha1/"

AGENT_SOURCES_VOLUME_MOUNT_PATH: str = "/data/user-agent-sources"
AGENT_SOURCES_CATALOGS_PATH_ARG: str = "--catalogs-path=/data/user-agent-sources/sources.yaml"

EXPECTED_DEFAULT_AGENT_CATALOG: dict = {
    "name": "Red Hat Agents",
    "id": "rh_agents",
    "type": "yaml",
    "enabled": True,
    "properties": {
        "yamlCatalogPath": "/shared-data/redhat-agents-catalog.yaml",
    },
    "labels": ["Red Hat Agents"],
}

EXPECTED_AGENT_LABEL_DEFINITION: dict = {
    "name": "Red Hat Agents",
    "assetType": "agents",
    "displayName": "Agent templates",
    "description": "Pre-built agent templates from the Red Hat agentic starter kits collection.",
}
