TEMPLATE_ARTIFACT_TYPE: str = "template-artifact"

ARTIFACT_FULL_AGENT_NAME: str = "artifact-test-agent"
ARTIFACT_EMPTY_AGENT_NAME: str = "artifact-empty-agent"
ARTIFACT_IMAGE_ONLY_AGENT_NAME: str = "artifact-image-only-agent"
ARTIFACT_DEFAULT_NAME_AGENT_NAME: str = "artifact-default-name-agent"
DEFAULT_TEMPLATE_NAME: str = "agent.yaml"
ARTIFACT_TEST_SOURCE_ID: str = "test_artifact_catalog"
ARTIFACT_TEST_SOURCE_NAME: str = "Test Artifact Catalog"
ARTIFACT_TEST_YAML_PATH: str = "artifact-agents.yaml"
ARTIFACT_TEST_LABEL: str = "Test Artifacts"

ARTIFACT_TEST_SOURCE: dict = {
    "name": ARTIFACT_TEST_SOURCE_NAME,
    "id": ARTIFACT_TEST_SOURCE_ID,
    "type": "yaml",
    "enabled": True,
    "properties": {"yamlCatalogPath": ARTIFACT_TEST_YAML_PATH},
    "labels": [ARTIFACT_TEST_LABEL],
}

ARTIFACT_TEST_LABEL_DEFINITION: dict = {
    "name": ARTIFACT_TEST_LABEL,
    "assetType": "agents",
    "displayName": "Test artifact agents",
}

ARTIFACT_IMAGE_COUNT: int = 6
ARTIFACT_TEMPLATE_COUNT: int = 4

EXPECTED_TEMPLATE_NAMES: set[str] = {"agent.yaml", "deploy.yaml", "service.yaml", "route.yaml"}
ARTIFACT_IMAGE_ONLY_COUNT: int = 3

ARTIFACT_TEST_AGENTS_YAML: str = """\
source: Test Artifact Catalog
agents:
  - name: artifact-test-agent
    displayName: Artifact Test Agent
    description: Agent with both image and template artifacts for testing.
    framework: langgraph
    artifacts:
      - uri: quay.io/test/agent-image:v1.0
      - uri: quay.io/test/agent-image:v2.0
      - uri: quay.io/test/agent-image:v3.0-gpu
      - uri: quay.io/test/agent-image:v3.0-cpu
      - uri: quay.io/test/agent-sidecar:v1.0
      - uri: ghcr.io/test/agent-tools:latest
    templates:
      - name: agent.yaml
        content: |
          name: artifact-test-agent
          framework: langgraph
          env:
            - name: API_KEY
              required: true
      - name: deploy.yaml
        content: |
          apiVersion: apps/v1
          kind: Deployment
          metadata:
            name: artifact-test-agent
      - name: service.yaml
        content: |
          apiVersion: v1
          kind: Service
          metadata:
            name: artifact-test-agent
      - name: route.yaml
        content: |
          apiVersion: route.openshift.io/v1
          kind: Route
          metadata:
            name: artifact-test-agent
    createTimeSinceEpoch: "1748736000000"
    lastUpdateTimeSinceEpoch: "1750550400000"

  - name: artifact-empty-agent
    displayName: Empty Artifact Agent
    description: Agent with no artifacts or templates.
    framework: crewai
    createTimeSinceEpoch: "1748736000000"
    lastUpdateTimeSinceEpoch: "1750550400000"

  - name: artifact-default-name-agent
    displayName: Default Name Agent
    description: Agent with a template that has no explicit name.
    framework: langgraph
    templates:
      - content: |
          name: artifact-default-name-agent
          framework: langgraph
    createTimeSinceEpoch: "1748736000000"
    lastUpdateTimeSinceEpoch: "1750550400000"

  - name: artifact-image-only-agent
    displayName: Image Only Agent
    description: Agent with only image artifacts and no templates.
    framework: autogen
    artifacts:
      - uri: quay.io/test/image-only:v1.0
      - uri: quay.io/test/image-only:v2.0
      - uri: quay.io/test/image-only:v3.0
    createTimeSinceEpoch: "1748736000000"
    lastUpdateTimeSinceEpoch: "1750550400000"
"""
