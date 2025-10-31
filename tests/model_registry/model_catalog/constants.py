from typing import Any

from tests.model_registry.constants import SAMPLE_MODEL_NAME1, CUSTOM_CATALOG_ID1

CUSTOM_CATALOG_ID2: str = "sample_custom_catalog2"

SAMPLE_MODEL_NAME2 = "mistralai/Devstral-Small-2505"
EXPECTED_CUSTOM_CATALOG_VALUES: list[dict[str, str]] = [{"id": CUSTOM_CATALOG_ID1, "model_name": SAMPLE_MODEL_NAME1}]
MULTIPLE_CUSTOM_CATALOG_VALUES: list[dict[str, str]] = [
    {"id": CUSTOM_CATALOG_ID1, "model_name": SAMPLE_MODEL_NAME1},
    {"id": CUSTOM_CATALOG_ID2, "model_name": SAMPLE_MODEL_NAME2},
]

REDHAT_AI_CATALOG_NAME: str = "Red Hat AI"
REDHAT_AI_VALIDATED_CATALOG_NAME: str = "Red Hat AI validated"
REDHAT_AI_VALIDATED_UNESCAPED_CATALOG_NAME: str = "Red Hat AI Validated"
REDHAT_AI_FILTER: str = "Red+Hat+AI"
REDHAT_AI_VALIDATED_FILTER = "Red+Hat+AI+Validated"

SAMPLE_MODEL_NAME3 = "mistralai/Ministral-8B-Instruct-2410"
CATALOG_CONTAINER: str = "catalog"
DEFAULT_CATALOGS: dict[str, Any] = {
    "redhat_ai_models": {
        "name": REDHAT_AI_CATALOG_NAME,
        "type": "yaml",
        "properties": {"yamlCatalogPath": "/shared-data/models-catalog.yaml"},
        "labels": [REDHAT_AI_CATALOG_NAME],
    },
    "redhat_ai_validated_models": {
        "name": REDHAT_AI_VALIDATED_CATALOG_NAME,
        "type": "yaml",
        "properties": {"yamlCatalogPath": "/shared-data/validated-models-catalog.yaml"},
        "labels": [REDHAT_AI_VALIDATED_CATALOG_NAME],
    },
}
REDHAT_AI_CATALOG_ID: str = "redhat_ai_models"
DEFAULT_CATALOG_FILE: str = DEFAULT_CATALOGS[REDHAT_AI_CATALOG_ID]["properties"]["yamlCatalogPath"]
VALIDATED_CATALOG_ID: str = "redhat_ai_validated_models"

MODEL_ARTIFACT_TYPE: str = "model-artifact"
METRICS_ARTIFACT_TYPE: str = "metrics-artifact"
PERFORMANCE_DATA_DIR: str = "/shared-benchmark-data"
