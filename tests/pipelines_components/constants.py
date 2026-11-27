import os
from pathlib import Path
from typing import Any

from tests.pipelines_components.image_constants import PipelinesComponentsImages


def _load_env_file(env_path: Path) -> None:
    """Parse a .env file and set variables into os.environ (does not overwrite existing)."""
    if not env_path.is_file():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key not in os.environ:
            os.environ[key] = value


_load_env_file(env_path=Path(__file__).parent / ".env")

# DSPA configuration
DSPA_NAME: str = "dspa"
DSPA_MINIO_IMAGE: str = os.getenv(
    "DSPA_MINIO_IMAGE",
    PipelinesComponentsImages.MINIO,
)
DSPA_PIPELINE_DEPLOYMENT: str = f"ds-pipeline-{DSPA_NAME}"
DSPA_SCHEDULED_WORKFLOW_DEPLOYMENT: str = f"ds-pipeline-scheduledworkflow-{DSPA_NAME}"
DSPA_S3_SECRET: str = f"ds-pipeline-s3-{DSPA_NAME}"
DSPA_S3_BUCKET: str = "mlpipeline"

# AutoML task configurations for parametrized testing
AUTOML_TASK_CONFIGS: dict[str, dict[str, Any]] = {
    "regression": {
        "label_column": "price",
        "task_type": "regression",
        "top_n": 1,
    },
    "classification": {
        "label_column": "target",
        "task_type": "binary",
        "top_n": 1,
    },
    "multiclass": {
        "label_column": "target",
        "task_type": "multiclass",
        "top_n": 1,
    },
}

# AutoML pipeline parameters — the smoke uploader prefixes this MinIO key with the task type
AUTOML_TRAIN_DATA_FILE_KEY: str = os.getenv("AUTOML_TRAIN_DATA_FILE_KEY", "automl-smoke/train.csv")

# AutoML timeseries configuration
AUTOML_TIMESERIES_CONFIG: dict[str, Any] = {
    "id_column": "series_id",
    "timestamp_column": "Month",
    "target": "Sunspots",
    "prediction_length": 1,
    "top_n": 3,
    "known_covariates_names": [],
}
# Timeouts (seconds)
AUTOML_PIPELINE_START_TIMEOUT: int = int(os.getenv("AUTOML_PIPELINE_START_TIMEOUT", "300"))
PIPELINE_POLL_INTERVAL: int = int(os.getenv("PIPELINE_POLL_INTERVAL", "30"))

MINIO_MC_IMAGE: str = os.getenv(
    "MINIO_MC_IMAGE",
    PipelinesComponentsImages.MINIO,
)
MINIO_UPLOADER_SECURITY_CONTEXT: dict[str, object] = {
    "allowPrivilegeEscalation": False,
    "capabilities": {"drop": ["ALL"]},
    "runAsNonRoot": True,
    "seccompProfile": {"type": "RuntimeDefault"},
}

# AutoRAG pipeline parameters
AUTORAG_INPUT_DATA_KEY: str = os.getenv("AUTORAG_INPUT_DATA_KEY", "autorag-smoke/input_data")
AUTORAG_TEST_DATA_KEY: str = os.getenv("AUTORAG_TEST_DATA_KEY", "autorag-smoke/benchmark_data.json")
AUTORAG_MAX_RAG_PATTERNS: int = int(os.getenv("AUTORAG_MAX_RAG_PATTERNS", "4"))
AUTORAG_OPTIMIZATION_METRIC: str = os.getenv("AUTORAG_OPTIMIZATION_METRIC", "unitxt:faithfulness")

AUTORAG_PIPELINE_START_TIMEOUT: int = int(os.getenv("AUTORAG_PIPELINE_START_TIMEOUT", "300"))

# ---------------------------------------------------------------------------
# Managed pipelines (DSPA operator auto-registers pipelines in KFP)
# ---------------------------------------------------------------------------
MANAGED_PIPELINE_AUTOML_TABULAR: str = os.getenv(
    "MANAGED_PIPELINE_AUTOML_TABULAR", "autogluon-tabular-training-pipeline"
)
MANAGED_PIPELINE_AUTOML_TIMESERIES: str = os.getenv(
    "MANAGED_PIPELINE_AUTOML_TIMESERIES", "autogluon-timeseries-training-pipeline"
)
MANAGED_PIPELINE_AUTORAG: str = os.getenv("MANAGED_PIPELINE_AUTORAG", "documents-rag-optimization-pipeline")
MANAGED_PIPELINES_IMAGE: str = os.getenv("MANAGED_PIPELINES_IMAGE", "")
DSPA_READY_BUFFER_SECONDS: int = int(os.getenv("DSPA_READY_BUFFER_SECONDS", "30"))
MANAGED_PIPELINE_WAIT_TIMEOUT: int = int(os.getenv("MANAGED_PIPELINE_WAIT_TIMEOUT", "300"))
MANAGED_PIPELINE_POLL_INTERVAL: int = int(os.getenv("MANAGED_PIPELINE_POLL_INTERVAL", "15"))
