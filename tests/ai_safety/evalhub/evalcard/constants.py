# EvalHub does not serve eval cards over its own API. On job completion it uploads the card
# to a new MLflow run (tagged with the job ID) in the job's experiment, and optionally pushes
# it to an OCI registry when the job has an `exports.oci` block.

# MLflow REST paths (relative to the MLflow base URL)
MLFLOW_RUNS_SEARCH_PATH: str = "/api/2.0/mlflow/runs/search"
MLFLOW_ARTIFACTS_PATH: str = "/api/2.0/mlflow-artifacts/artifacts"
MLFLOW_WORKSPACE_HEADER: str = "X-MLFLOW-WORKSPACE"

# MLflow artifact name and run tag written by EvalHub
EVALCARD_MLFLOW_ARTIFACT_NAME: str = "evaluation-card.json"
EVALCARD_MLFLOW_RUN_TAG_KEY: str = "evaluation_job_id"

# OCI export constants. EvalHub pushes the manifest as "<oci_tag>-<job_id>".
EVALCARD_OCI_REPO: str = "evalhub/evalcard"
EVALCARD_OCI_TAG: str = "v1"
EVALCARD_OCI_LAYER_MEDIA_TYPE: str = "application/vnd.eval-hub.evaluation-card.v1+json"

# EvalCard version constants
EVALCARD_SCHEMA_VERSION: str = "1.0"
EVALCARD_CARD_VERSION: str = "1.0"

# Structural validation: required keys at each level
EVALCARD_REQUIRED_TOP_LEVEL_KEYS: set[str] = {
    "card_version",
    "schema_version",
    "metadata",
    "context",
    "results",
}

EVALCARD_METADATA_REQUIRED_KEYS: set[str] = {
    "evaluation_job_id",
    "created_at",
    "updated_at",
}

# Collection jobs carry context.collection_id instead of context.benchmarks
EVALCARD_CONTEXT_REQUIRED_KEYS: set[str] = {
    "model",
}

EVALCARD_RESULTS_REQUIRED_KEYS: set[str] = {
    "benchmarks",
}

EVALCARD_BENCHMARK_RESULT_REQUIRED_KEYS: set[str] = {
    "status",
    "metrics",
}
