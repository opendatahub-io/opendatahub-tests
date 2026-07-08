# Custom test catalog injected via ConfigMap patch (mirrors MCP_CATALOG_SOURCE pattern).
TEST_AGENT_CATALOG_SOURCE_ID: str = "test_agent_catalog"
TEST_AGENT_CATALOG_SOURCE_NAME: str = "Test Agent Catalog"
TEST_AGENTS_YAML_CATALOG_PATH: str = "agents-catalog.yaml"
TEST_AGENT_CATALOG_LABEL: str = "Test Agents"

REQUIRED_AGENT_FIELDS: list[str] = ["name", "displayName", "description", "framework"]

# Minimum filter fields required by filterQuery tests (subset of API response).
EXPECTED_FILTER_OPTIONS: set[str] = {
    "framework",
}

LANGGRAPH_FRAMEWORK: str = "langgraph"
CREWAI_FRAMEWORK: str = "crewai"
AUTOGEN_FRAMEWORK: str = "autogen"
CLAUDE_CODE_FRAMEWORK: str = "claude-code"

# Agent counts and names used by the test ConfigMap patch.
TEST_AGENT_COUNT: int = 5
TEST_LANGGRAPH_AGENT_COUNT: int = 2
TEST_LANGGRAPH_AGENT_NAMES: set[str] = {"langgraph-react-agent", "langgraph-agentic-rag"}
TEST_AGENT_NAMES: set[str] = {
    "langgraph-react-agent",
    "langgraph-agentic-rag",
    "crewai-websearch-agent",
    "autogen-mcp-agent",
    "claude-code",
}

TEST_AGENT_CATALOG_SOURCE: dict = {
    "name": TEST_AGENT_CATALOG_SOURCE_NAME,
    "id": TEST_AGENT_CATALOG_SOURCE_ID,
    "type": "yaml",
    "enabled": True,
    "properties": {"yamlCatalogPath": TEST_AGENTS_YAML_CATALOG_PATH},
    "labels": [TEST_AGENT_CATALOG_LABEL],
}

TEST_AGENT_LABEL_DEFINITION: dict = {
    "name": TEST_AGENT_CATALOG_LABEL,
    "assetType": "agents",
    "displayName": "Test agent starter kits",
}

# Minimal agent catalog YAML injected by the test ConfigMap patch.
TEST_AGENTS_YAML: str = """\
source: Test Agent Catalog
agents:
  - name: langgraph-react-agent
    displayName: LangGraph ReAct Agent
    description: General-purpose ReAct agent built with LangGraph using a reason-and-act loop with external tools.
    framework: langgraph
    labels:
      - react
      - tools
    createTimeSinceEpoch: "1748736000000"
    lastUpdateTimeSinceEpoch: "1750550400000"

  - name: langgraph-agentic-rag
    displayName: LangGraph Agentic RAG
    description: Agentic RAG agent built with LangGraph for retrieval-augmented workflows.
    framework: langgraph
    labels:
      - rag
      - retrieval
    createTimeSinceEpoch: "1748736000000"
    lastUpdateTimeSinceEpoch: "1750550400000"

  - name: crewai-websearch-agent
    displayName: CrewAI Web Search Agent
    description: Web search agent built with CrewAI.
    framework: crewai
    labels:
      - web-search
    createTimeSinceEpoch: "1747267200000"
    lastUpdateTimeSinceEpoch: "1750550400000"

  - name: autogen-mcp-agent
    displayName: AutoGen MCP Agent
    description: MCP-enabled agent built with Microsoft AutoGen.
    framework: autogen
    labels:
      - mcp
    createTimeSinceEpoch: "1745798400000"
    lastUpdateTimeSinceEpoch: "1750550400000"

  - name: claude-code
    displayName: Claude Code on OpenShift
    description: Deploy Claude Code on OpenShift with multiple backend options.
    framework: claude-code
    labels:
      - deployment
    createTimeSinceEpoch: "1746835200000"
    lastUpdateTimeSinceEpoch: "1750550400000"
"""
