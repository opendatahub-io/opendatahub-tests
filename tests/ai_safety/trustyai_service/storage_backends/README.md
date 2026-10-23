# TrustyAI SQL storage tests

Every TrustyAI service in this suite is created from a `TrustyAIService` custom
resource. The operator creates and owns the service Deployment, Route, image,
serving TLS, and database wiring. The tests do not create a TrustyAI Deployment
or select a service image. PostgreSQL Deployments are separate test database
resources.

The CR storage format remains `DATABASE`. The database Secret referenced by
`spec.storage.databaseConfigurations` selects the SQL backend through
`databaseKind` (`postgresql` or `sqlite`). PostgreSQL tests use a TLS-enabled
PostgreSQL service, a CA Secret named `<TrustyAIService name>-db-ca`, and the
operator-created TrustyAI service Route with its normal bearer token.

## Coverage

| Behavior | Test |
| --- | --- |
| `DATABASE` CR plus `databaseKind=postgresql` reaches a PostgreSQL server over TLS | `test_postgres_storage.py` |
| Upload, metadata, append, inference IDs, name mapping, and tags | `test_postgres_storage.py` |
| Readiness, health, concurrent writes, and PostgreSQL data after service-pod replacement | `test_storage_health_and_pooling.py` |
| Operator Secret references, PostgreSQL selector, JDBC TLS settings, and CA mount | `test_database_configuration.py` |
| Valid CA, no-CA default, wrong-CA rejection, missing configured CA, and explicit insecure opt-in | `test_postgres_tls.py` |
| PostgreSQL pool/session settings; operator-side (CR condition) and service-side (pod log) invalid setting rejection; SQLite pragmas | `test_db_config_map.py` |
| SQLite API round trip, configured file path, and in-memory data loss after process replacement | `test_sqlite_storage.py` |

PostgreSQL tuning tests provide the tuning ConfigMap through
`spec.storage.databaseTuning`; the operator imports those values into the
CR-managed service pod. The suite updates the ConfigMap after the service is
running and restarts its pod so the new environment reaches the process. It
checks live PostgreSQL session and pool settings, invalid-value rejection, and
normal API behavior with tuning enabled. SQLite pragma settings
also come through this CR reference.

Invalid tuning values are rejected in two places. The operator validates value
format (non-negative integers, the `DATABASE_SYNCHRONOUS_COMMIT` enum, SQLite
mode enums, a non-blank application name) and reports `DatabaseTuningAvailable=False`
with reason `InvalidConfiguration` on the CR, naming only the first bad key in
sorted order, before it touches the pod. The service rejects what the operator
accepts: `DATABASE_POOL_SIZE=0` and a pool ceiling above
`DATABASE_MAX_CONNECTIONS_BUDGET`; those show up in the pod log and the pod
stays unready.

SQLite is tested as a single-pod service backend selected through the referenced
database Secret. The service uses its default `:memory:` database; SQLite
pragmas are supplied through the tuning ConfigMap. SQLite should not be used as
a multi-replica persistent backend.

These tests cover service pod replacement with PostgreSQL data preserved; they
do not cover a PostgreSQL image or schema upgrade. There is no direct TrustyAI
service image validation in this suite: the image comes from the cluster's
installed operator configuration.

## Running

```bash
uv run pytest tests/ai_safety/trustyai_service/storage_backends -m ai_safety
```
