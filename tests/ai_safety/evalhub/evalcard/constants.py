# EvalCard API endpoint (update when finalized)
EVALHUB_EVALCARD_PATH_TEMPLATE: str = "/api/v1/evaluations/jobs/{job_id}/evalcard"

# EvalCard version constants
EVALCARD_SCHEMA_VERSION: str = "1.0"
EVALCARD_CARD_VERSION: str = "1.0"

# MLflow artifact name
EVALCARD_MLFLOW_ARTIFACT_NAME: str = "evalcard.yaml"

# OCI export constants
EVALCARD_OCI_REPO: str = "evalhub/evalcard"
EVALCARD_OCI_TAG: str = "v1"

# Structural validation: required keys at each level
EVALCARD_REQUIRED_TOP_LEVEL_KEYS: set[str] = {
    "card_version",
    "schema_version",
    "metadata",
    "context",
    "results",
    "references",
}

EVALCARD_METADATA_REQUIRED_KEYS: set[str] = {
    "evaluation_job_id",
    "created_at",
    "updated_at",
}

EVALCARD_CONTEXT_REQUIRED_KEYS: set[str] = {
    "model",
    "benchmarks",
}

EVALCARD_RESULTS_REQUIRED_KEYS: set[str] = {
    "benchmarks",
}

EVALCARD_BENCHMARK_RESULT_REQUIRED_KEYS: set[str] = {
    "status",
    "metrics",
}
