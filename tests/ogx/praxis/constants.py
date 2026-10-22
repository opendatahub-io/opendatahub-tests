"""Constants shared by the OGX -> Praxis migration tests."""

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

# Third target table. Written by the migration but not compared by the parity
# test, which asserts on responses and conversations only.
TARGET_CONVERSATION_ITEMS_TABLE: str = "openai_conversation_items"

# --- Praxis gateway deployed to own the target schema ---
PRAXIS_GATEWAY_APP_LABEL: str = "praxis-gateway"
PRAXIS_GATEWAY_POD_LABELS: dict[str, str] = {"app": PRAXIS_GATEWAY_APP_LABEL}
PRAXIS_GATEWAY_DEPLOYMENT_NAME: str = "praxis-gateway-deployment"
PRAXIS_GATEWAY_SERVICE_NAME: str = "praxis-gateway-service"
PRAXIS_GATEWAY_CONFIG_SECRET_NAME: str = "praxis-gateway-config"
PRAXIS_GATEWAY_CONFIG_KEY: str = "config.yaml"
PRAXIS_GATEWAY_CONFIG_MOUNT_PATH: str = "/etc/praxis"
PRAXIS_GATEWAY_CONTAINER_NAME: str = "praxis"
PRAXIS_GATEWAY_PORT: int = 8080
PRAXIS_GATEWAY_READY_TIMEOUT: int = 300

# Tenancy the migration Job is given for source rows that carry none. The Praxis
# gateway is configured with the same tenant so the rows the migration writes and
# the ones Praxis would serve belong to one identity.
MIGRATION_FALLBACK_TENANT: str = "test-tenant"
MIGRATION_FALLBACK_OWNER_SUBJECT: str = "test-owner"

# Secrets holding the source and target connection strings. The operator's
# migration preflight requires both Secrets to carry the `ogx.io/watch` label.
OGX_CONNECTION_SECRET_NAME: str = "ogx-db-connection"
OGX_CONNECTION_SECRET_KEY: str = "connection-string"
PRAXIS_CONNECTION_SECRET_NAME: str = "praxis-db-connection"
PRAXIS_CONNECTION_SECRET_KEY: str = "connection-string"

# Shared by both PostgreSQL instances, which come from the same pod template.
POSTGRES_CONTAINER_NAME: str = "postgres"
POSTGRES_PORT: int = 5432

# Dummy data seeded before the upgrade.
SEED_RESPONSES_COUNT: int = 3
SEED_CONVERSATIONS_COUNT: int = 2
SEED_RESPONSE_MAX_OUTPUT_TOKENS: int = 64
SEED_MARKER: str = "praxis-migration-upgrade"
