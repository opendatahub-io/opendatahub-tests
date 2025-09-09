CUSTOM_CATALOG_ID1: str = "sample_custom_catalog1"
CUSTOM_CATALOG_ID2: str = "sample_custom_catalog2"
SAMPLE_MODEL_NAME1 = "mistralai/Mistral-7B-Instruct-v0.3"

SAMPLE_MODEL_NAME2 = "mistralai/Devstral-Small-2505"
EXPECTED_CUSTOM_CATALOG_VALUES: list[dict[str, str]] = [{"id": CUSTOM_CATALOG_ID1, "model_name": SAMPLE_MODEL_NAME1}]
MULTIPLE_CUSTOM_CATALOG_VALUES: list[dict[str, str]] = [
    {"id": CUSTOM_CATALOG_ID1, "model_name": SAMPLE_MODEL_NAME1},
    {"id": CUSTOM_CATALOG_ID2, "model_name": SAMPLE_MODEL_NAME2},
]
DEFAULT_CATALOG_NAME: str = "Default Catalog"
DEFAULT_CATALOG_ID: str = "default_catalog"
CATALOG_TYPE: str = "yaml"
DEFAULT_CATALOG_FILE: str = "/default/default-catalog.yaml"
SAMPLE_MODEL_NAME3 = "mistralai/Ministral-8B-Instruct-2410"
