ALL_ARTIFACT_CATEGORIES: set[str] = {"model-artifact", "performance-metrics", "accuracy-metrics"}

NO_ARTIFACT_CATALOG_ID: str = "no_artifact_test_catalog"
NO_ARTIFACT_CATALOG_YAML_FILENAME: str = "no-artifact-test-catalog.yaml"
NO_ARTIFACT_MODELS: list[str] = ["test-org/fake-model-alpha", "test-org/fake-model-beta"]
NO_ARTIFACT_SOURCES_YAML: str = f"""\
catalogs:
  - name: No Artifact Test Catalog
    id: {NO_ARTIFACT_CATALOG_ID}
    type: yaml
    enabled: true
    properties:
      yamlCatalogPath: {NO_ARTIFACT_CATALOG_YAML_FILENAME}
"""
NO_ARTIFACT_YAML: str = """\
source: test
models:
  - name: test-org/fake-model-alpha
    description: Model with no artifacts
    provider: Test
    createTimeSinceEpoch: "1700000000000"
    lastUpdateTimeSinceEpoch: "1700000000000"
  - name: test-org/fake-model-beta
    description: Another model with no artifacts
    provider: Test
    createTimeSinceEpoch: "1700000000000"
    lastUpdateTimeSinceEpoch: "1700000000000"
"""

STATUS_PATH_TEMPLATE: str = "{base}sources/{source_id}/status"
UNKNOWN_SOURCE_ID: str = "does_not_exist_xyz"
ERROR_SOURCE_ID: str = "mixed_models_catalog"
ERROR_SOURCE_YAML: str = f"""
catalogs:
  - name: Source Status Error Catalog
    id: {ERROR_SOURCE_ID}
    type: hf
    enabled: true
    includedModels:
      - "jonburdo/public-test-model-1"
      - "jonburdo/private-test-model-1"
    labels:
      - {ERROR_SOURCE_ID}
"""
