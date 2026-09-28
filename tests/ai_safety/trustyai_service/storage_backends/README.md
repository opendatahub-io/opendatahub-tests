# TrustyAI service SQL storage backend tests

Cluster coverage for the SQL storage work in the Python `trustyai-service`
(`RHAISTRAT-2662`): the shared SQLAlchemy Core `SQLStorage` base and the
PostgreSQL and SQLite backends built on it, plus the TLS posture, health checks
and connection pooling that came with them.

## Why these tests do not use a `TrustyAIService` CR

The operator cannot select the new backends today:

* `api/tas/v1/trustyaiservice_types.go` restricts `spec.storage.format` with
  `+kubebuilder:validation:Enum=PVC;DATABASE`.
* `controllers/tas/templates/service/deployment.tmpl.yaml` copies that value
  straight into `SERVICE_STORAGE_FORMAT`.
* The service maps `DATABASE` to MariaDB; PostgreSQL needs `POSTGRESQL` (or
  `POSTGRES`) and SQLite needs `SQLITE`.

So a CR-driven test could only ever exercise MariaDB. These tests therefore
deploy the service image directly as a `Deployment` with the env the backend
actually reads, which is the only way to exercise the new code on a cluster
until the operator grows a PostgreSQL value. When it does, the fixtures here
stay useful: only the "how is the service deployed" fixture changes.

## Why a plain PostgreSQL `Deployment` and not a PostgreSQL operator

An operator (Crunchy Postgres for Kubernetes, CloudNativePG) is installable —
`utilities/infra.install_operator` already does this for `mariadb-operator` in
`tests/conftest.py::installed_mariadb_operator` — but for this suite it costs
more than it returns:

| | plain `Deployment` | operator |
| --- | --- | --- |
| Availability | image is already pinned as `SharedImages.POSTGRESQL_15` | needs an OperatorHub catalog, so it is unavailable on disconnected clusters |
| Setup cost | seconds | CSV install, minutes, per run |
| Blast radius | namespace-scoped | cluster-scoped install shared with every other suite |
| TLS | self-signed CA generated in-test (already needed, see below) | operator-issued certs |

The one real operator advantage — certificates for free — does not apply here,
because the suite already generates a CA to prove `sslmode=verify-full`, and
the same generator is shared with the existing MariaDB fixtures. Every other
component in this repo (model registry, OGX, autorag) also runs PostgreSQL as a
plain `Deployment`.

## Reused from elsewhere in the repo

* `tests/ai_safety/trustyai_service/utils.py::generate_db_tls_certs` — the
  certificate generator extracted from the existing MariaDB fixture so both
  databases share one implementation.
* `utilities.image_constants.SharedImages.POSTGRESQL_15` — the same pinned
  image the OGX and autorag suites use.
* `tests/ai_safety/conftest.py::trustyai_operator_configmap` — source of the
  `trustyaiServiceImage` under test, so the suite always runs the image the
  installed operator would deploy.
* `utilities.certificates_utils.create_ca_bundle_file` — router CA bundle for
  verified HTTPS calls to the service route.
* `tests/conftest.py::model_namespace`, `teardown_resources` — standard
  namespace lifecycle.

## Layout

| File | Contents |
| --- | --- |
| `constants.py` | names, ports, env keys, backend ids |
| `utils.py` | PostgreSQL deployment, service deployment, HTTP client, log/readiness helpers |
| `conftest.py` | fixtures for the database, credentials, CA secret and each backend flavour |
| `test_postgres_storage.py` | PostgreSQL end-to-end through the public API |
| `test_postgres_tls.py` | TLS enforcement and the insecure opt-out |
| `test_sqlite_storage.py` | SQLite backend, including `:memory:` semantics |
| `test_storage_health_and_pooling.py` | health checks, pool sizing, concurrent writes, restart persistence |

## Coverage map

| Behaviour under test | Source | Test |
| --- | --- | --- |
| `SERVICE_STORAGE_FORMAT=POSTGRESQL` selects `PostgreSQLStorage` | `storage/__init__.py:258` | `test_postgres_storage.py::TestPostgresStorageBackend::test_service_reports_postgres_backend` |
| write/read round trip through `SQLStorage` | `storage/sql/base.py` | `test_upload_creates_datasets` |
| metadata: shape, column names, row counts | `storage/sql/base.py` | `test_metadata_reports_uploaded_shape` |
| dynamic per-dataset tables over repeated uploads | `storage/sql/schema.py` | `test_repeated_uploads_append_rows` |
| inference id listing | `endpoints/metadata.py:261` | `test_inference_ids_listed` |
| name mapping apply and clear | `endpoints/metadata.py:456,524` | `test_name_mapping_round_trip` |
| tag listing | `endpoints/metadata.py:663` | `test_tags_reported_for_uploaded_data` |
| data survives a pod restart (real persistence, not process state) | — | `test_storage_health_and_pooling.py::test_data_survives_service_restart` |
| CA required by default, error names the CA path | `storage/__init__.py:47,193` | `test_postgres_tls.py::test_service_refuses_to_start_without_ca` |
| `DATABASE_ALLOW_INSECURE_TLS` is the only opt-out | `storage/__init__.py:25` | `test_insecure_opt_in_allows_startup` |
| `sslmode=verify-full` against the mounted CA | `storage/sql/engine.py:89` | `test_postgres_storage.py` (all tests run over verified TLS) |
| `SERVICE_STORAGE_FORMAT=SQLITE` + `STORAGE_DATABASE_PATH` | `storage/__init__.py:211` | `test_sqlite_storage.py` |
| SQLite `:memory:` is not durable | `storage/sql/engine.py:132` | `test_memory_backend_does_not_persist_across_restart` |
| readiness reports the SQL backend | `service/health_checks.py:180` | `test_storage_health_and_pooling.py::test_readiness_reports_storage_backend` |
| pool sizing env vars are honoured | `storage/sql/engine.py:36` | `test_pool_env_is_applied` |
| concurrent writers do not exhaust or deadlock the pool | `storage/sql/engine.py:42` | `test_concurrent_uploads_share_the_pool` |

## How the suite reaches the service

`main.py::run_server` keeps the plain-HTTP API on `127.0.0.1:8081`, because in a
CR-managed deployment kube-rbac-proxy forwards to it. A Service pointed at 8081
therefore never connects. The process only opens a routable listener —
`0.0.0.0:4443` — when it finds serving certificates at `/etc/tls/internal`.

So the Service carries the
`service.beta.openshift.io/serving-cert-secret-name` annotation, the container
mounts the resulting secret, and the Route is `reencrypt` with the namespace's
service-CA bundle as `destinationCACertificate`. Health probes stay on
`0.0.0.0:8080`, which the process always exposes.

Verified against a locally built image (`podman build --platform linux/amd64`):
without certificates, hypercorn ignores `HTTP_PORT` and serves the API on its
own default, `127.0.0.1:8000`, since `insecure_bind` only takes effect when TLS
is configured. Mounting the certificates avoids that path entirely.

## Prerequisites

The service image must ship `sqlalchemy` (all SQL backends) and `psycopg`
(PostgreSQL). `requirements.txt` in `trustyai-service` was compiled with
`--extra mariadb --extra mmd` only, which left out both — an image built from it
could not run any SQL backend, MariaDB included, since `maria/maria.py` now
imports the shared SQLAlchemy base. It has since been recompiled with
`--extra postgres`; confirm any image under test was built after that.

Verified on a local `linux/amd64` build of the recompiled requirements: SQLite,
PostgreSQL and MariaDB all report `Storage readiness: ok` and round-trip
upload, `/info`, `/info/tags`, inference ids and name mapping.

## Running

```bash
uv run pytest tests/ai_safety/trustyai_service/storage_backends -m ai_safety
```
