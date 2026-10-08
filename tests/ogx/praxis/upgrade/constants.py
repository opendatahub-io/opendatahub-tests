"""Constants for the OGX -> Praxis database migration tests."""

from typing import Literal

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

# Secret holding the Praxis connection string. The upgrade step must point
# `spec.praxisMode.migrationJob.targetConnectionString` at this name and key;
# the tests create the Secret before the upgrade so it is there to be referenced.
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
