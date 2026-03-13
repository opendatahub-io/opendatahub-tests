from typing import Any

from tests.model_registry.constants import CUSTOM_CATALOG_ID1, SAMPLE_MODEL_NAME1

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
        "enabled": True,
    },
    "redhat_ai_validated_models": {
        "name": REDHAT_AI_VALIDATED_CATALOG_NAME,
        "type": "yaml",
        "properties": {"yamlCatalogPath": "/shared-data/validated-models-catalog.yaml"},
        "labels": [REDHAT_AI_VALIDATED_CATALOG_NAME],
        "enabled": True,
    },
}
REDHAT_AI_CATALOG_ID: str = "redhat_ai_models"
DEFAULT_CATALOG_FILE: str = DEFAULT_CATALOGS[REDHAT_AI_CATALOG_ID]["properties"]["yamlCatalogPath"]
VALIDATED_CATALOG_ID: str = "redhat_ai_validated_models"
VALIDATED_CATALOG_FILE: str = DEFAULT_CATALOGS[VALIDATED_CATALOG_ID]["properties"]["yamlCatalogPath"]

MODEL_ARTIFACT_TYPE: str = "model-artifact"
METRICS_ARTIFACT_TYPE: str = "metrics-artifact"
PERFORMANCE_DATA_DIR: str = "/shared-benchmark-data"
HF_SOURCE_ID: str = "huggingface_mixed"
HF_MODEL_NAME: str = "ibm-granite/granite-speech-3.2-8b"
# TODO: get a service account to host these models
HF_CUSTOM_MODE: str = "jonburdo/test2"
HF_MODELS: dict[str, Any] = {
    "mixed": [
        # Generative models (text-generation)
        "ibm-granite/granite-4.0-h-1b",
        "microsoft/phi-2",
        "microsoft/Phi-4-mini-reasoning",
        "microsoft/Phi-3.5-mini-instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        # Predictive models (text-classification)
        "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
        # Predictive models (image-classification)
        "google/vit-base-patch16-224",
        # Potential unknown models (base models or unusual tasks)
        "sentence-transformers/all-MiniLM-L6-v2",  # sentence-similarity (not in lists)
        "openai/clip-vit-base-patch32",  # zero-shot-image-classification (not in lists)
    ],
    "granite": [
        "ibm-granite/granite-4.0-h-small",
        "ibm-granite/granite-4.0-micro",
        "ibm-granite/granite-4.0-h-350m",
        "ibm-granite/granite-4.0-micro-base",
        "ibm-granite/granite-4.0-h-micro",
    ],
    "custom": [HF_CUSTOM_MODE],
}
EXPECTED_HF_CATALOG_VALUES: list[dict[str, str]] = [{"id": HF_SOURCE_ID, "model_name": HF_MODELS["mixed"][0]}]
EXPECTED_MULTIPLE_HF_CATALOG_VALUES: list[dict[str, str]] = [
    {"id": HF_SOURCE_ID, "model_name": HF_MODELS["mixed"][0]},
    {"id": "huggingface_granite", "model_name": HF_MODELS["granite"][0]},
]
