import os
from pathlib import Path
from typing import Any


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
    "quay.io/opendatahub/minio:RELEASE.2019-08-14T20-37-41Z-license-compliance",
)
DSPA_PIPELINE_DEPLOYMENT: str = f"ds-pipeline-{DSPA_NAME}"
DSPA_SCHEDULED_WORKFLOW_DEPLOYMENT: str = f"ds-pipeline-scheduledworkflow-{DSPA_NAME}"
DSPA_S3_SECRET: str = f"ds-pipeline-s3-{DSPA_NAME}"
DSPA_S3_BUCKET: str = "mlpipeline"

# Pipeline YAML paths — provided via .env or environment variables
AUTOML_PIPELINE_YAML: str = os.environ.get("AUTOML_PIPELINE_YAML", "")

# AutoML S3 source — CSV is downloaded from this external S3 path into DSPA MinIO
AUTOML_S3_BUCKET: str = os.environ.get("AUTOML_S3_BUCKET", "")

# AutoML task configurations for parametrized testing
AUTOML_TASK_CONFIGS: dict[str, dict[str, Any]] = {
    "regression": {
        "s3_train_data_key": "datasets/regression/regression.csv",
        "label_column": "price",
        "task_type": "regression",
        "top_n": 1,
    },
    "classification": {
        "s3_train_data_key": "datasets/classification/classification.csv",
        "label_column": "target",
        "task_type": "binary",
        "top_n": 1,
    },
    "multiclass": {
        "s3_train_data_key": "datasets/classification/multiclass.csv",
        "label_column": "target",
        "task_type": "multiclass",
        "top_n": 1,
    },
}

# AutoML pipeline parameters — AUTOML_TRAIN_DATA_FILE_KEY is the destination key in DSPA MinIO
AUTOML_TRAIN_DATA_FILE_KEY: str = os.getenv("AUTOML_TRAIN_DATA_FILE_KEY", "automl-smoke/train.csv")

# Timeouts (seconds)
AUTOML_PIPELINE_TIMEOUT: int = int(os.getenv("AUTOML_PIPELINE_TIMEOUT", "1800"))
PIPELINE_POLL_INTERVAL: int = int(os.getenv("PIPELINE_POLL_INTERVAL", "30"))

MINIO_MC_IMAGE: str = os.getenv(
    "MINIO_MC_IMAGE",
    "quay.io/minio/mc@sha256:470f5546b596e16c7816b9c3fa7a78ce4076bb73c2c73f7faeec0c8043923123",
)
MINIO_UPLOADER_SECURITY_CONTEXT: dict[str, object] = {
    "allowPrivilegeEscalation": False,
    "capabilities": {"drop": ["ALL"]},
    "runAsNonRoot": True,
    "seccompProfile": {"type": "RuntimeDefault"},
}

AUTORAG_PIPELINE_YAML: str = os.environ.get("AUTORAG_PIPELINE_YAML", "")

AUTORAG_S3_BUCKET: str = os.environ.get("AUTORAG_S3_BUCKET", "mlpipeline")

# LlamaStack catalog-compatible model ID for the inference model (optional).
# The rh-dev LlamaStack distribution validates INFERENCE_MODEL against its model catalog (Meta Llama,
# IBM Granite, etc.).  If AUTORAG_INFERENCE_MODEL_NAME is not a catalog model (e.g. Qwen2.5-0.5B-Instruct),
# set this to a supported catalog ID (e.g. meta-llama/Llama-3.2-1B-Instruct).  LlamaStack will register
# that catalog model but route inference calls to vLLM using AUTORAG_INFERENCE_MODEL_NAME as the
# provider_model_id, so the actual weights served by vLLM are used regardless of the catalog name.
# Defaults to AUTORAG_INFERENCE_MODEL_NAME when unset (works as-is if the name is catalog-compatible).
AUTORAG_LLAMA_STACK_INFERENCE_MODEL_ID: str = os.environ.get("AUTORAG_LLAMA_STACK_INFERENCE_MODEL_ID", "")

AUTORAG_EMBEDDING_MAX_MODEL_LEN: str = os.getenv("AUTORAG_EMBEDDING_MAX_MODEL_LEN", "512")

# AutoRAG pipeline parameters
AUTORAG_INPUT_DATA_KEY: str = os.getenv("AUTORAG_INPUT_DATA_KEY", "autorag-smoke/input_data")
AUTORAG_TEST_DATA_KEY: str = os.getenv("AUTORAG_TEST_DATA_KEY", "autorag-smoke/benchmark_data.json")
AUTORAG_MAX_RAG_PATTERNS: int = int(os.getenv("AUTORAG_MAX_RAG_PATTERNS", "4"))
AUTORAG_OPTIMIZATION_METRIC: str = os.getenv("AUTORAG_OPTIMIZATION_METRIC", "faithfulness")

# AutoRAG timeouts (seconds)
AUTORAG_PIPELINE_TIMEOUT: int = int(os.getenv("AUTORAG_PIPELINE_TIMEOUT", "3600"))

# ---------------------------------------------------------------------------
# Managed pipelines (DSPA operator auto-registers pipelines in KFP)
# ---------------------------------------------------------------------------
MANAGED_PIPELINE_AUTOML_TABULAR: str = os.getenv(
    "MANAGED_PIPELINE_AUTOML_TABULAR", "autogluon-tabular-training-pipeline"
)
MANAGED_PIPELINE_AUTORAG: str = os.getenv("MANAGED_PIPELINE_AUTORAG", "documents-rag-optimization-pipeline")
MANAGED_PIPELINES_IMAGE: str = os.getenv("MANAGED_PIPELINES_IMAGE", "")
DSPA_READY_BUFFER_SECONDS: int = int(os.getenv("DSPA_READY_BUFFER_SECONDS", "30"))
MANAGED_PIPELINE_WAIT_TIMEOUT: int = int(os.getenv("MANAGED_PIPELINE_WAIT_TIMEOUT", "300"))
MANAGED_PIPELINE_POLL_INTERVAL: int = int(os.getenv("MANAGED_PIPELINE_POLL_INTERVAL", "15"))
