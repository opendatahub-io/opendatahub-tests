import os
from pathlib import Path


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

# AutoML pipeline parameters
AUTOML_TRAIN_DATA_FILE_KEY: str = os.getenv("AUTOML_TRAIN_DATA_FILE_KEY", "automl-smoke/train.csv")
AUTOML_LABEL_COLUMN: str = os.getenv("AUTOML_LABEL_COLUMN", "target")
AUTOML_TASK_TYPE: str = os.getenv("AUTOML_TASK_TYPE", "binary")
AUTOML_TOP_N: int = int(os.getenv("AUTOML_TOP_N", "1"))

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

# AutoRAG pipeline parameters
AUTORAG_INPUT_DATA_KEY: str = os.getenv("AUTORAG_INPUT_DATA_KEY", "autorag-smoke/input_data")
AUTORAG_TEST_DATA_KEY: str = os.getenv("AUTORAG_TEST_DATA_KEY", "autorag-smoke/benchmark_data.json")
AUTORAG_MAX_RAG_PATTERNS: int = int(os.getenv("AUTORAG_MAX_RAG_PATTERNS", "4"))
AUTORAG_OPTIMIZATION_METRIC: str = os.getenv("AUTORAG_OPTIMIZATION_METRIC", "faithfulness")

# AutoRAG timeouts (seconds)
AUTORAG_PIPELINE_TIMEOUT: int = int(os.getenv("AUTORAG_PIPELINE_TIMEOUT", "3600"))

# Synthetic CSV for AutoML binary classification smoke test.
# AutoGluon requires enough rows per class for stratified train/validation splits.
AUTOML_SMOKE_CSV: str = """feature_1,feature_2,feature_3,target
0.64,0.03,0.28,0
0.23,0.73,0.67,0
0.88,0.1,0.42,1
0.04,0.22,0.51,0
0.04,0.2,0.65,0
0.54,0.23,0.59,1
0.8,0.02,0.8,1
0.69,0.34,0.16,0
0.95,0.34,0.1,1
0.1,0.84,0.6,0
0.8,0.73,0.54,1
0.96,0.38,0.55,1
0.82,0.62,0.85,1
0.58,0.7,0.05,0
0.23,0.29,0.09,0
0.24,0.11,0.28,0
0.63,0.37,0.37,0
0.22,0.27,0.93,1
0.65,0.61,0.18,0
0.72,0.17,0.38,1
0.98,0.64,0.56,1
0.68,0.84,0.77,1
0.23,0.04,0.32,0
0.27,0.22,0.93,1
0.87,0.32,0.65,1
0.4,0.91,0.46,0
0.27,0.25,0.56,0
0.27,0.58,0.89,1
0.4,0.22,0.99,1
0.51,0.1,0.06,0
0.12,0.62,0.79,0
0.42,0.07,0.38,0
0.99,0.53,0.96,1
0.85,0.02,0.72,1
0.68,0.54,0.27,0
0.64,0.12,0.44,1
0.45,0.94,0.87,1
0.27,0.5,0.19,0
0.9,0.86,0.3,1
0.64,0.61,0.16,0
0.76,0.54,0.77,1
0.53,0.01,0.33,0
0.03,0.92,0.87,0
0.83,0.31,0.07,0
0.87,0.94,0.09,0
0.49,0.08,0.76,1
0.76,0.14,0.48,1
0.55,0.27,0.86,1
0.42,0.22,0.54,0
0.73,0.21,0.32,1
0.99,0.65,0.44,1
0.52,0.13,0.23,0
0.34,0.59,0.24,0
0.23,0.08,0.63,0
0.23,0.9,0.85,1
0.08,0.24,0.67,0
0.22,0.14,0.93,1
0.57,0.47,0.78,1
0.8,0.2,0.1,0
0.43,0.43,0.47,0
0.72,0.67,0.97,1
0.11,0.4,0.34,0
0.85,0.25,0.2,1
0.45,0.42,0.28,0
0.25,0.91,0.44,0
0.85,0.55,0.06,0
0.99,0.83,0.96,1
0.92,0.84,0.17,1
0.49,0.22,0.4,0
0.07,0.38,0.98,1
0.27,0.78,0.46,0
0.42,0.95,0.99,1
0.55,0.71,0.16,0
0.3,0.96,0.58,0
0.54,0.74,0.07,0
0.58,0.5,0.85,1
0.16,0.95,0.09,0
0.19,0.59,0.67,0
0.24,0.13,0.88,1
0.25,0.59,0.62,0
0.42,0.58,0.52,0
0.93,0.21,0.71,1
0.24,0.4,0.67,0
0.3,0.32,0.75,1
0.08,0.46,0.99,1
0.99,0.08,0.22,1
0.27,0.92,0.87,1
0.87,0.37,0.16,1
0.83,0.7,0.61,1
0.98,0.65,0.02,0
0.81,0.3,0.66,1
0.93,0.14,0.12,1
0.11,0.55,0.28,0
0.6,0.71,0.21,0
0.63,0.27,0.49,1
0.9,0.84,0.1,0
0.43,0.28,0.01,0
0.77,0.63,0.27,1
0.74,0.55,0.43,1
0.02,0.08,0.88,0
0.9,0.54,0.83,1
0.58,0.16,0.13,0
0.31,0.89,0.79,1
0.85,0.89,0.22,1
0.25,0.11,0.77,1
0.88,0.41,0.62,1
0.16,0.92,0.86,1
0.97,0.8,0.87,1
0.03,0.73,0.34,0
0.92,0.8,0.86,1
0.8,0.27,0.78,1
0.12,0.86,0.85,0
0.23,0.81,0.46,0
0.31,0.79,0.23,0
0.03,0.2,0.33,0
0.86,0.96,0.28,1
0.64,0.4,0.97,1
0.54,0.93,0.12,0
0.96,0.18,0.95,1
0.27,0.12,0.44,0
"""
