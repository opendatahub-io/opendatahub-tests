"""Constants shared by the OGX -> Praxis migration tests."""

from typing import Any, Literal

# Paths of the OpenAI-compatible APIs, as declared by the Gateway API routes
# that expose them after the migration to Praxis.
RESPONSES_API_PATH: str = "/v1/responses"
EMBEDDINGS_API_PATH: str = "/v1/embeddings"
MODELS_API_PATH: str = "/v1/models"
FILES_API_PATH: str = "/v1/files"
VECTOR_STORES_API_PATH: str = "/v1/vector_stores"
CONVERSATIONS_API_PATH: str = "/v1/conversations"

# Service port OGX listens on for traffic delegated by Praxis. In Praxis-fronted mode the
# operator-managed NetworkPolicy admits this port only from Praxis and the operator namespace.
OGX_DELEGATION_PORT: int = 8321

# Pod labels identifying Praxis, used as the OGXServer praxisSelector.
PRAXIS_POD_LABELS: dict[str, str] = {"app": "payload-processing"}

# Timeout for requests sent through the external Gateway hostname.
REQUEST_TIMEOUT_SECONDS: int = 30
PROBE_TIMEOUT_SECONDS: int = 30

# Parameters for the namespace the tests deploy into.
NAMESPACE_PARAMS: dict[str, Any] = {"name": "test-ogx-to-praxis"}

# Parameters for the OGXServer the tests deploy.
OGX_SERVER_PARAMS: dict[str, Any] = {
    "ogx_storage_size": "2Gi",
    "vector_io_provider": "pgvector",
    "files_provider": "local",
}

# Parameters for an OGXServer in Praxis-fronted internal-only mode. External access is
# requested on purpose: in Praxis mode the operator must not honor it.
PRAXIS_MODE_OGX_SERVER_PARAMS: dict[str, Any] = {
    **OGX_SERVER_PARAMS,
    "network": {"externalAccess": {"enabled": True}},
    "praxis_mode": {
        "enabled": True,
        "praxisSelector": {"podSelector": {"matchLabels": PRAXIS_POD_LABELS}},
    },
}

# Source (OGX) tables. Both live in the OGX PostgreSQL database deployed by the
# `postgres_deployment` fixture. The names come from the built-in config of the
# distribution the tests deploy: no `providers`/`storage`/`overrideConfig` is set
# on the OGXServer, so the operator generates no config to override them and
# `ogx migrate praxis` reads exactly these tables. Update them if the
# distribution renames its tables. They are interpolated into SQL, so they must
# stay plain unquoted identifiers and must never become externally settable.
SOURCE_RESPONSES_TABLE: str = "agents_responses"
SOURCE_CONVERSATIONS_TABLE: str = "openai_conversations"

# Target (Praxis) tables. These are the `ogx migrate praxis` defaults; the
# migration Job created by the operator does not override them.
TARGET_RESPONSES_TABLE: str = "openai_responses"
TARGET_CONVERSATIONS_TABLE: str = "openai_conversations"

# Tables holding the Files and Vector Stores metadata, read by the write-path
# ownership test. Unlike the tables above these are not part of the migration's
# contract -- `ogx migrate praxis` does not read them -- so nothing in this repo
# or in the distribution config pins their names.
#
# TODO: confirm all four names against a live cluster.
SOURCE_FILES_TABLE: str = "openai_files"
SOURCE_VECTOR_STORES_TABLE: str = "openai_vector_stores"
TARGET_FILES_TABLE: str = "openai_files"
TARGET_VECTOR_STORES_TABLE: str = "openai_vector_stores"

# OGX PostgreSQL instance deployed in the test namespace (see
# `build_ogx_server_config` and the `postgres_deployment` fixture).
OGX_POSTGRES_DATABASE: str = "ps_db"
OGX_POSTGRES_POD_LABEL_SELECTOR: str = "app=postgres"

# Praxis (target) PostgreSQL instance, deployed by these tests from the same
# template as the OGX one. The app label differs so the two pod lookups cannot
# match each other's pod, and the database differs because the operator's
# migration preflight refuses to migrate a database onto itself.
PRAXIS_POSTGRES_APP_LABEL: str = "praxis-postgres"
PRAXIS_POSTGRES_POD_LABEL_SELECTOR: str = f"app={PRAXIS_POSTGRES_APP_LABEL}"
PRAXIS_POSTGRES_DEPLOYMENT_NAME: str = "praxis-postgres-deployment"
PRAXIS_POSTGRES_SERVICE_NAME: str = "praxis-postgres-service"
PRAXIS_POSTGRES_DATABASE: str = "praxis_db"

# Secrets holding the source and target connection strings. The operator's
# migration preflight requires both Secrets to carry the `ogx.io/watch` label.
OGX_CONNECTION_SECRET_NAME: str = "ogx-db-connection"
OGX_CONNECTION_SECRET_KEY: str = "connection-string"
PRAXIS_CONNECTION_SECRET_NAME: str = "praxis-db-connection"
PRAXIS_CONNECTION_SECRET_KEY: str = "connection-string"

# Shared by both PostgreSQL instances, which come from the same pod template.
POSTGRES_CONTAINER_NAME: str = "postgres"
POSTGRES_PORT: int = 5432

# Migration Job created by the OGX operator: `<ogxserver-name>-praxis-migration`.
MIGRATION_JOB_NAME_SUFFIX: str = "-praxis-migration"
MIGRATION_JOB_TIMEOUT: int = 600

# Dummy data seeded before the upgrade.
SEED_RESPONSES_COUNT: int = 3
SEED_CONVERSATIONS_COUNT: int = 2
SEED_RESPONSE_MAX_OUTPUT_TOKENS: int = 64
SEED_MARKER: str = "praxis-migration-upgrade"

# Files seeded through the Files API before the upgrade, whose ids must still
# resolve unchanged afterwards.
SEED_FILES_COUNT: int = 3
# Narrowed to the literal the Files API accepts, so the value stays assignable
# to `FilesResource.create(purpose=...)`.
SEED_FILE_PURPOSE: Literal["assistants"] = "assistants"

# ConfigMap carrying the pre-upgrade Files/Vector Stores API responses into the
# post-upgrade run, following the pattern used by the MaaS upgrade tests.
API_BASELINE_CONFIG_MAP_NAME: str = "praxis-upgrade-api-baseline"
API_BASELINE_CONFIG_MAP_KEY: str = "api_baseline"

# Section of the same ConfigMap holding the file_search citation inputs. Each
# pre-upgrade test writes its own key, so tests sharing a namespace do not
# overwrite each other's baseline.
FILE_SEARCH_CITATIONS_CONFIG_MAP_KEY: str = "file_search_citations"

# Inputs for the file_search citation test. The question is answered by the
# IBM 2025 Q4 earnings release, the single document the pre-upgrade run ingests
# into its vector store.
CITATION_QUESTION: str = "How did IBM perform financially in the fourth quarter of 2025?"
CITATION_INSTRUCTIONS: str = "Always use the file_search tool to look up information before answering."
CITATION_MAX_OUTPUT_TOKENS: int = 512

# Response fields compared byte-for-byte across the upgrade. The Files set is the
# one named by the test case; vector stores have no `bytes`/`filename`, so their
# equivalents are compared instead.
COMPARED_FILE_FIELDS: tuple[str, ...] = ("id", "bytes", "filename", "created_at", "status")
COMPARED_VECTOR_STORE_FIELDS: tuple[str, ...] = ("id", "name", "created_at", "status")

# Equivalent sets for the two remaining resource kinds the state inventory samples.
COMPARED_RESPONSE_FIELDS: tuple[str, ...] = ("id", "created_at", "model", "status")
COMPARED_CONVERSATION_FIELDS: tuple[str, ...] = ("id", "created_at")

# --- Rollback of external routing from Praxis back to OGX ---
#
# The rollback procedure itself is applied to the test environment between the
# pre- and post-rollback runs, the same way the upgrade is, so nothing here
# describes how it is triggered.

# Service the OGX operator creates for an OGXServer: `<ogxserver-name>-service`.
# Its pods are the ones whose logs must show the post-rollback request.
OGX_SERVICE_NAME_SUFFIX: str = "-service"

# Response created through the external hostname after the rollback, to be
# correlated against the OGX pod logs. It is deliberately unstored, so that
# probing does not change the response count the state inventory compares.
ROLLBACK_PROBE_MARKER: str = "praxis-rollback-probe"

# Section of the shared baseline ConfigMap holding the pre-rollback state inventory.
ROLLBACK_INVENTORY_CONFIG_MAP_KEY: str = "rollback_inventory"

# Upper bound on the rows an inventory listing walks. Reached only when the
# distribution holds far more state than these tests create, which would make the
# counts meaningless rather than merely slow, so it fails instead of truncating.
MAX_LISTED_RESOURCES: int = 1000
