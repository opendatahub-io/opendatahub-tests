import os

from utilities.path_utils import resolve_repo_path

# DSPA configuration
DSPA_NAME: str = "dspa"
DSPA_MINIO_IMAGE: str = "quay.io/opendatahub/minio:RELEASE.2019-08-14T20-37-41Z-license-compliance"
DSPA_PIPELINE_DEPLOYMENT: str = f"ds-pipeline-{DSPA_NAME}"
DSPA_SCHEDULED_WORKFLOW_DEPLOYMENT: str = f"ds-pipeline-scheduledworkflow-{DSPA_NAME}"
DSPA_S3_SECRET: str = f"ds-pipeline-s3-{DSPA_NAME}"
DSPA_S3_BUCKET: str = "mlpipeline"

# Pipeline YAML — pre-compiled from red-hat-data-services/pipelines-components
_AUTOML_DEFAULT_YAML: str = str(
    resolve_repo_path(source="tests/pipelines_components/automl/pipelines/autogluon_tabular_training.yaml")
)
AUTOML_PIPELINE_YAML: str = os.getenv("AUTOML_PIPELINE_YAML", _AUTOML_DEFAULT_YAML)

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

# Synthetic CSV for AutoML binary classification smoke test.
# AutoGluon requires enough rows per class for stratified train/validation splits.
AUTOML_SMOKE_CSV: str = """feature_1,feature_2,feature_3,target
0.12,0.45,0.78,0
0.91,0.23,0.56,1
0.34,0.67,0.89,0
0.78,0.12,0.34,1
0.56,0.89,0.12,0
0.23,0.56,0.91,1
0.67,0.34,0.45,0
0.45,0.78,0.67,1
0.89,0.91,0.23,0
0.11,0.11,0.56,1
0.33,0.44,0.77,0
0.77,0.33,0.44,1
0.55,0.88,0.11,0
0.22,0.55,0.99,1
0.66,0.22,0.33,0
0.44,0.77,0.66,1
0.88,0.66,0.22,0
0.99,0.99,0.88,1
0.15,0.35,0.55,0
0.85,0.15,0.75,1
0.25,0.75,0.95,0
0.75,0.25,0.15,1
0.35,0.85,0.35,0
0.65,0.65,0.85,1
0.95,0.05,0.65,0
0.05,0.95,0.05,1
0.41,0.59,0.41,0
0.59,0.41,0.59,1
0.71,0.29,0.71,0
0.29,0.71,0.29,1
"""
