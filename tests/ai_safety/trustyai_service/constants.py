from typing import Any

from tests.ai_safety.image_constants import AiSafetyImages

DRIFT_BASE_DATA_PATH: str = "./tests/ai_safety/trustyai_service/drift/model_data"
TAI_DATA_CONFIG: dict[str, str] = {"filename": "data.csv", "format": "CSV"}
TAI_METRICS_CONFIG: dict[str, str] = {"schedule": "5s"}
TAI_PVC_STORAGE_CONFIG: dict[str, str] = {"format": "PVC", "folder": "/inputs", "size": "1Gi"}
TAI_DB_STORAGE_CONFIG: dict[str, str] = {
    "format": "DATABASE",
    "size": "1Gi",
    "databaseConfigurations": "db-credentials",
}

MLSERVER: str = "mlserver"
MLSERVER_RUNTIME_NAME: str = f"{MLSERVER}-1.x"
XGBOOST: str = "xgboost"
LIGHTGBM: str = "lightgbm"
MLFLOW: str = "mlflow"

GAUSSIAN_CREDIT_MODEL: str = "gaussian-credit-model"
GAUSSIAN_CREDIT_MODEL_STORAGE_URI: str = AiSafetyImages.GAUSSIAN_CREDIT_MODEL
GAUSSIAN_CREDIT_MODEL_RESOURCES: dict[str, dict[str, str]] = {
    "requests": {"cpu": "1", "memory": "500Mi"},
    "limits": {"cpu": "1", "memory": "500Mi"},
}

KSERVE_MLSERVER: str = f"kserve-{MLSERVER}"

ISVC_GETTER: str = "isvc-getter"

TRUSTYAI_DB_MIGRATION_PATCH: dict[str, Any] = {
    "metadata": {"annotations": {"trustyai.opendatahub.io/db-migration": "true"}},
    "spec": {
        "storage": {
            "format": "DATABASE",
            "folder": "/inputs",
            "size": "1Gi",
            "databaseConfigurations": "db-credentials",
        },
        "data": {"filename": "data.csv", "format": "BEAN"},
    },
}
