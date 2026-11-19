"""Constants for IBM CLEAR adapter tests (RHOAIENG-63558).

Covers direct MLflow trace ingestion and PVC/file-staged agent traces for
agentic evaluation via provider ``ibm-clear``.
"""

from tests.ai_safety.image_constants import AiSafetyImages

IBM_CLEAR_PROVIDER_ID: str = "ibm-clear"
IBM_CLEAR_PROVIDER_NAME: str = "IBM CLEAR FVT"
CLEAR_BENCHMARK_ID: str = "agentic-evaluation"
CLEAR_BENCHMARK_CUSTOM_CRITERIA_ID: str = "agentic-evaluation-custom-criteria"
CLEAR_BENCHMARK_PREDEFINED_ISSUES_ID: str = "agentic-evaluation-predefined-issues"

CLEAR_ADAPTER_IMAGE: str = AiSafetyImages.CLEAR_ADAPTER
CLEAR_ADAPTER_ENTRYPOINT: list[str] = ["python", "main.py"]

CLEAR_EVALHUB_CR_NAME: str = "evalhub-mt"
MLFLOW_SERVICE_PORT: int = 8443
MLFLOW_STATIC_PREFIX: str = "/mlflow"

# Input-trace experiment (traces fetched by CLEAR) vs results experiment (artifacts uploaded).
CLEAR_TRACES_EXPERIMENT_NAME: str = "odh-fvt-clear-agent-traces"
CLEAR_RESULTS_EXPERIMENT_NAME: str = "odh-fvt-clear-agent-results"
CLEAR_UNKNOWN_EXPERIMENT_NAME: str = "odh-fvt-clear-experiment-does-not-exist"

CLEAR_JOB_TIMEOUT: int = 1800
CLEAR_JOB_POLL_INTERVAL: int = 30

CLEAR_EXPECTED_METRICS: tuple[str, ...] = (
    "total_interactions",
    "total_issues",
    "interactions_with_issues",
    "interactions_no_issues",
    "total_agents",
    "pct_interactions_with_issues",
    "issues_per_interaction",
    "average_score",
)

CLEAR_TRACES_PVC_NAME: str = "evalhub-clear-traces"
CLEAR_TRACES_PVC_SIZE: str = "1Gi"
CLEAR_TRACES_SUB_PATH: str = "traces"
CLEAR_TRACE_FILENAME: str = "tr-fvt-minimal-agent.json"

# MLflow REST paths used by seed helpers (ODH tracking URI includes /mlflow prefix).
MLFLOW_EXPERIMENTS_CREATE_PATH: str = "/api/2.0/mlflow/experiments/create"
MLFLOW_EXPERIMENTS_GET_BY_NAME_PATH: str = "/api/2.0/mlflow/experiments/get-by-name"
MLFLOW_TRACES_SEARCH_PATH: str = "/api/2.0/mlflow/traces/search"
MLFLOW_TRACES_V3_PATH: str = "/api/3.0/mlflow/traces"

MLFLOW_WORKSPACE_HEADER: str = "X-MLFLOW-WORKSPACE"
