from typing import Any

from utilities.constants import ApiGroups, Ports

DRIFT_BASE_DATA_PATH: str = "./tests/model_explainability/trustyai_service/drift/model_data"
TAI_DATA_CONFIG: dict[str, str] = {"filename": "data.csv", "format": "CSV"}
TAI_METRICS_CONFIG: dict[str, str] = {"schedule": "5s"}
TAI_PVC_STORAGE_CONFIG: dict[str, str] = {"format": "PVC", "folder": "/inputs", "size": "1Gi"}
TAI_DB_STORAGE_CONFIG: dict[str, str] = {
    "format": "DATABASE",
    "size": "1Gi",
    "databaseConfigurations": "db-credentials",
}

SKLEARN: str = "sklearn"
MLSERVER: str = "mlserver"
MLSERVER_RUNTIME_NAME: str = f"{MLSERVER}-1.x"
XGBOOST: str = "xgboost"
LIGHTGBM: str = "lightgbm"
MLFLOW: str = "mlflow"

GAUSSIAN_CREDIT_MODEL: str = "gaussian-credit-model"
GAUSSIAN_CREDIT_MODEL_STORAGE_PATH: str = f"{SKLEARN}/{GAUSSIAN_CREDIT_MODEL.replace('-', '_')}/1"
GAUSSIAN_CREDIT_MODEL_RESOURCES: dict[str, dict[str, str]] = {
    "requests": {"cpu": "1", "memory": "2Gi"},
    "limits": {"cpu": "1", "memory": "2Gi"},
}

KSERVE_MLSERVER: str = f"kserve-{MLSERVER}"
KSERVE_MLSERVER_SUPPORTED_MODEL_FORMATS: list[dict[str, Any]] = [
    {"name": "sklearn", "version": "0", "autoSelect": True, "priority": 2},
    {"name": "sklearn", "version": "1", "autoSelect": True, "priority": 2},
    {"name": "xgboost", "version": "1", "autoSelect": True, "priority": 2},
    {"name": "xgboost", "version": "2", "autoSelect": True, "priority": 2},
    {"name": "lightgbm", "version": "3", "autoSelect": True, "priority": 2},
    {"name": "lightgbm", "version": "4", "autoSelect": True, "priority": 2},
    {"name": "mlflow", "version": "1", "autoSelect": True, "priority": 1},
    {"name": "mlflow", "version": "2", "autoSelect": True, "priority": 1},
]
KSERVE_MLSERVER_CONTAINERS: list[dict[str, Any]] = [
    {
        "name": "kserve-container",
        "image": "quay.io/trustyai_testing/mlserver"
        "@sha256:68a4cd74fff40a3c4f29caddbdbdc9e54888aba54bf3c5f78c8ffd577c3a1c89",
        "env": [
            {"name": "MLSERVER_MODEL_IMPLEMENTATION", "value": "{{.Labels.modelClass}}"},
            {"name": "MLSERVER_HTTP_PORT", "value": str(Ports.REST_PORT)},
            {"name": "MLSERVER_GRPC_PORT", "value": "9000"},
            {"name": "MODELS_DIR", "value": "/mnt/models/"},
        ],
        "resources": {"requests": {"cpu": "1", "memory": "2Gi"}, "limits": {"cpu": "1", "memory": "2Gi"}},
    }
]
KSERVE_MLSERVER_ANNOTATIONS: dict[str, str] = {
    f"{ApiGroups.OPENDATAHUB_IO}/accelerator-name": "",
    f"{ApiGroups.OPENDATAHUB_IO}/template-display-name": "KServe MLServer",
    "prometheus.kserve.io/path": "/metrics",
    "prometheus.io/port": str(Ports.REST_PORT),
    "openshift.io/display-name": "mlserver-1.x",
}

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
