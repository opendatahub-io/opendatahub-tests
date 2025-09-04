CUSTOM_ECHO_CATALOG_ID: str = "sample_rhec_catalog"
CUSTOM_MODEL_NAME = "rhelai1/modelcar-granite-7b-starter"
CUSTOM_ECOSYSTEM_CATALOG: str = f"""catalogs:
- name: Red Hat Ecosystem Catalog
  id: {CUSTOM_ECHO_CATALOG_ID}
  type: rhec
  enabled: true
  properties:
    models:
    - {CUSTOM_MODEL_NAME}
"""
EXPECTED_ECHO_CATALOG_VALUES: dict[str, str] = {
    "id": CUSTOM_ECHO_CATALOG_ID,
    "model_name": f"{CUSTOM_MODEL_NAME}:latest",
}
CUSTOM_CATALOG_ID: str = "sample_custom_catalog"
CUSTOM_CATALOG_WITH_FILE = f"""catalogs:
- name: Sample Catalog
  id: {CUSTOM_CATALOG_ID}
  type: yaml
  enabled: true
  properties:
    yamlCatalogPath: sample-catalog.yaml
"""
SAMPLE_CATALOG_FILE_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
SAMPLE_CATALOG_YAML = f"""source: Hugging Face
models:
- name: {SAMPLE_CATALOG_FILE_NAME}
  description: test description.
  readme: |-
    # test read me information
  provider: Mistral AI
  logo: temp placeholder logo
  license: apache-2.0
  licenseLink: https://www.apache.org/licenses/LICENSE-2.0.txt
  libraryName: transformers
  artifacts:
    - uri: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/resolve/main/consolidated.safetensors
"""
EXPECTED_CUSTOM_CATALOG_VALUES: dict[str, str] = {"id": CUSTOM_CATALOG_ID, "model_name": SAMPLE_CATALOG_FILE_NAME}
DEFAULT_CATALOG_NAME: str = "Default Catalog"
DEFAULT_CATALOG_ID: str = "default_catalog"
CATALOG_TYPE: str = "yaml"
DEFAULT_CATALOG_FILE: str = "/default/default-catalog.yaml"
