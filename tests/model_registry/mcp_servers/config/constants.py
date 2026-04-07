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

CALCULATOR_SERVER_NAME: str = "calculator"
CALCULATOR_PROVIDER: str = "Math Community"
MCP_CATALOG_SOURCE_ID: str = "test_mcp_servers"
MCP_CATALOG_SOURCE_NAME: str = "Test MCP Servers"
MCP_SERVERS_YAML_CATALOG_PATH: str = "mcp-servers.yaml"

MCP_SERVERS_YAML: str = """\
mcp_servers:
  - name: weather-api
    description: "Community weather API MCP server"
    provider: "Weather Community"
    version: "1.0.0"
    license: "MIT"
    tags:
      - weather
      - api
      - community
    tools:
      - name: get_current_weather
        description: "Get current weather for a location"
      - name: get_forecast
        description: "Get weather forecast"
    createTimeSinceEpoch: "1736510400000"
    lastUpdateTimeSinceEpoch: "1736510400000"

  - name: file-manager
    description: "File system management MCP server"
    provider: "Community Dev"
    version: "0.9.2"
    license: "BSD-3-Clause"
    tags:
      - filesystem
      - files
      - management
    tools:
      - name: read_file
        description: "Read file contents"
      - name: write_file
        description: "Write to files"
      - name: list_directory
        description: "List directory contents"
    createTimeSinceEpoch: "1738300800000"
    lastUpdateTimeSinceEpoch: "1739510400000"

  - name: calculator
    description: "Mathematical calculator MCP server"
    provider: "Math Community"
    version: "2.0.0"
    license: "MIT"
    tags:
      - math
      - calculator
      - computation
    customProperties:
      verifiedSource:
        metadataType: MetadataBoolValue
        bool_value: true
      sast:
        metadataType: MetadataBoolValue
        bool_value: true
      readOnlyTools:
        metadataType: MetadataBoolValue
        bool_value: true
      observability:
        metadataType: MetadataStringValue
        string_value: ""
    tools:
      - name: calculate
        description: "Perform mathematical calculations"
      - name: solve_equation
        description: "Solve mathematical equations"
    createTimeSinceEpoch: "1740091200000"
    lastUpdateTimeSinceEpoch: "1740091200000"
"""

MCP_CATALOG_SOURCE: dict = {
    "name": MCP_CATALOG_SOURCE_NAME,
    "id": MCP_CATALOG_SOURCE_ID,
    "type": "yaml",
    "enabled": True,
    "properties": {"yamlCatalogPath": MCP_SERVERS_YAML_CATALOG_PATH},
    "labels": [MCP_CATALOG_SOURCE_NAME],
}


MCP_CATALOG_SOURCE2_ID: str = "test_mcp_servers_2"
MCP_CATALOG_SOURCE2_NAME: str = "Test MCP Servers 2"
MCP_SERVERS_YAML2_CATALOG_PATH: str = "mcp-servers-2.yaml"

MCP_SERVERS_YAML2: str = """\
mcp_servers:
  - name: code-reviewer
    description: "Code review assistant MCP server"
    provider: "DevOps Tools"
    version: "1.2.0"
    license: "Apache-2.0"
    tags:
      - code
      - review
    tools:
      - name: review_pull_request
        description: "Review a pull request"
      - name: suggest_fix
        description: "Suggest a code fix"
"""

MCP_CATALOG_SOURCE2: dict = {
    "name": MCP_CATALOG_SOURCE2_NAME,
    "id": MCP_CATALOG_SOURCE2_ID,
    "type": "yaml",
    "enabled": True,
    "properties": {"yamlCatalogPath": MCP_SERVERS_YAML2_CATALOG_PATH},
    "labels": [MCP_CATALOG_SOURCE2_NAME],
}

EXPECTED_MCP_SERVER_NAMES: set[str] = {"weather-api", "file-manager", "calculator"}

EXPECTED_MCP_SERVER_TOOL_COUNTS: dict[str, int] = {
    "weather-api": 2,
    "file-manager": 3,
    "calculator": 2,
}

EXPECTED_MCP_SERVER_TOOLS: dict[str, list[str]] = {
    "weather-api": ["get_current_weather", "get_forecast"],
    "file-manager": ["read_file", "write_file", "list_directory"],
    "calculator": ["calculate", "solve_equation"],
}

EXPECTED_MCP_SERVER_TIMESTAMPS: dict[str, dict[str, str]] = {
    "weather-api": {"createTimeSinceEpoch": "1736510400000", "lastUpdateTimeSinceEpoch": "1736510400000"},
    "file-manager": {"createTimeSinceEpoch": "1738300800000", "lastUpdateTimeSinceEpoch": "1739510400000"},
    "calculator": {"createTimeSinceEpoch": "1740091200000", "lastUpdateTimeSinceEpoch": "1740091200000"},
}

MCP_CATALOG_INVALID_SOURCE_ID: str = "test_mcp_servers_invalid"
MCP_CATALOG_INVALID_SOURCE_NAME: str = "Test MCP Servers Invalid"
MCP_SERVERS_YAML_INVALID_CATALOG_PATH: str = "mcp-servers-invalid.yaml"

MCP_SERVERS_YAML_MALFORMED: str = """\
mcp_servers:
  - name: broken-server
    description: "This YAML has a syntax error
    version: "1.0.0"
  - name: [invalid
"""

MCP_SERVERS_YAML_MISSING_NAME: str = """\
mcp_servers:
  - description: "Server without a name field"
    provider: "Unnamed Provider"
    version: "1.0.0"
    tools:
      - name: some_tool
        description: "A tool on a nameless server"
"""

MCP_CATALOG_INVALID_SOURCE: dict = {
    "name": MCP_CATALOG_INVALID_SOURCE_NAME,
    "id": MCP_CATALOG_INVALID_SOURCE_ID,
    "type": "yaml",
    "enabled": True,
    "properties": {"yamlCatalogPath": MCP_SERVERS_YAML_INVALID_CATALOG_PATH},
    "labels": [MCP_CATALOG_INVALID_SOURCE_NAME],
}

NAMED_QUERIES: dict = {
    "production_ready": {
        "verifiedSource": {"operator": "=", "value": True},
    },
    "security_focused": {
        "sast": {"operator": "=", "value": True},
        "readOnlyTools": {"operator": "=", "value": True},
    },
}

EXPECTED_MCP_SOURCE2_SERVER_NAMES: set[str] = {"code-reviewer"}
EXPECTED_ALL_MCP_SERVER_NAMES: set[str] = EXPECTED_MCP_SERVER_NAMES | EXPECTED_MCP_SOURCE2_SERVER_NAMES

EXPECTED_MCP_SOURCE_ID_MAP: dict[str, str] = {
    "weather-api": MCP_CATALOG_SOURCE_ID,
    "file-manager": MCP_CATALOG_SOURCE_ID,
    "calculator": MCP_CATALOG_SOURCE_ID,
    "code-reviewer": MCP_CATALOG_SOURCE2_ID,
}

# Source 3: unlabeled source (no labels) for sourceLabel=null testing
MCP_CATALOG_SOURCE3_ID: str = "test_mcp_servers_unlabeled"
MCP_CATALOG_SOURCE3_NAME: str = "Test MCP Servers Unlabeled"
MCP_SERVERS_YAML3_CATALOG_PATH: str = "mcp-servers-3.yaml"

MCP_SERVERS_YAML3: str = """\
mcp_servers:
  - name: database-connector
    description: "Database connection MCP server"
    provider: "Data Tools"
    version: "1.0.0"
    license: "PostgreSQL"
    tags:
      - database
      - sql
    tools:
      - name: execute_query
        description: "Execute a database query"
"""

MCP_CATALOG_SOURCE3: dict = {
    "name": MCP_CATALOG_SOURCE3_NAME,
    "id": MCP_CATALOG_SOURCE3_ID,
    "type": "yaml",
    "enabled": True,
    "properties": {"yamlCatalogPath": MCP_SERVERS_YAML3_CATALOG_PATH},
}

EXPECTED_MCP_SOURCE3_SERVER_NAMES: set[str] = {"database-connector"}
EXPECTED_ALL_MCP_SERVER_NAMES_WITH_UNLABELED: set[str] = (
    EXPECTED_ALL_MCP_SERVER_NAMES | EXPECTED_MCP_SOURCE3_SERVER_NAMES
)
