"""Constants for the TrustyAI SQL storage backend tests."""

# Database deployed for the suite.
POSTGRES_SERVICE_NAME: str = "trustyai-postgres"
POSTGRES_PORT: int = 5432
POSTGRES_DB_NAME: str = "trustyai_db"
POSTGRES_DB_USERNAME: str = "trustyai_user"
POSTGRES_DB_PASSWORD: str = "trustyai_password"  # pragma: allowlist secret
POSTGRES_CREDENTIALS_SECRET_NAME: str = "postgres-db-credentials"
POSTGRES_CA_SECRET_NAME: str = "trustyai-db-ca"

# Where the PostgreSQL image reads extra server config and where we mount its certificates.
POSTGRES_CONFIG_MOUNT_PATH: str = "/opt/app-root/src/postgresql-cfg"
POSTGRES_CERTS_MOUNT_PATH: str = "/opt/app-root/src/certs"

# Service under test, deployed directly (see README.md for why there is no CR).
# Keep short so OpenShift auto-generated Route hostnames stay within the 63-char label limit.
STORAGE_SERVICE_NAME: str = "tai-sb"

# The plain-HTTP API binds 127.0.0.1 only (main.py keeps it loopback for
# kube-rbac-proxy), so the suite reaches the service over its HTTPS port, which
# the process opens on 0.0.0.0 once serving certificates are mounted.
SERVICE_TLS_PORT: int = 4443
SERVICE_TLS_MOUNT_PATH: str = "/etc/tls/internal"
SERVICE_CA_CONFIGMAP: str = "openshift-service-ca.crt"
SERVICE_CA_KEY: str = "service-ca.crt"
SERVING_CERT_ANNOTATION: str = "service.beta.openshift.io/serving-cert-secret-name"

# One deployment name per backend flavour, so a namespace can host several at once.
POSTGRES_SERVICE_DEPLOYMENT: str = f"{STORAGE_SERVICE_NAME}-postgres"
SQLITE_SERVICE_DEPLOYMENT: str = f"{STORAGE_SERVICE_NAME}-sqlite"
SQLITE_MEMORY_SERVICE_DEPLOYMENT: str = f"{STORAGE_SERVICE_NAME}-sqlite-memory"
SERVICE_HEALTH_PORT: int = 8080

# Mount point the service defaults to for the database CA (DEFAULT_TLS_CA_CERT).
SERVICE_DB_CA_MOUNT_PATH: str = "/etc/tls/db"
SERVICE_DB_CA_FILE: str = f"{SERVICE_DB_CA_MOUNT_PATH}/ca.crt"

# SQLite database location when the backend is file-backed.
SQLITE_DATA_MOUNT_PATH: str = "/var/lib/trustyai"
SQLITE_DATA_FILE: str = f"{SQLITE_DATA_MOUNT_PATH}/trustyai.db"
SQLITE_MEMORY_PATH: str = ":memory:"

# SERVICE_STORAGE_FORMAT values.
STORAGE_FORMAT_POSTGRES: str = "POSTGRESQL"
STORAGE_FORMAT_SQLITE: str = "SQLITE"

# Env vars the storage layer reads.
ENV_STORAGE_FORMAT: str = "SERVICE_STORAGE_FORMAT"
ENV_ALLOW_INSECURE_TLS: str = "DATABASE_ALLOW_INSECURE_TLS"
ENV_TLS_CA_CERT: str = "DATABASE_TLS_CA_CERT"
ENV_POOL_SIZE: str = "DATABASE_POOL_SIZE"
ENV_MAX_OVERFLOW: str = "DATABASE_MAX_OVERFLOW"
ENV_SQLITE_PATH: str = "STORAGE_DATABASE_PATH"

# Endpoints under test.
ENDPOINT_INFO: str = "info"
ENDPOINT_INFO_NAMES: str = "info/names"
ENDPOINT_INFO_TAGS: str = "info/tags"
ENDPOINT_INFERENCE_IDS: str = "info/inference/ids"
ENDPOINT_DATA_UPLOAD: str = "data/upload"
ENDPOINT_HEALTH: str = "q/health"
ENDPOINT_HEALTH_READY: str = "q/health/ready"

# Health check payload values, as emitted by service/health_checks.py.
HEALTH_STATUS_OK: str = "ok"
STORAGE_CHECK_NAME: str = "Storage readiness"
