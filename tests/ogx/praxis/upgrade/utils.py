"""Helpers for the OGX -> Praxis database migration tests.

The migration is performed by a Kubernetes Job (``<ogxserver>-praxis-migration``)
that the OGX operator creates from ``spec.praxisMode.migrationJob``. The Job runs
``ogx migrate praxis``, which reads the OGX PostgreSQL tables and writes them into
the Praxis PostgreSQL database referenced by ``targetConnectionString``.

Both databases run as PostgreSQL pods in the test namespace, so these helpers
read them by exec'ing ``psql`` inside the respective pod and parity can be
asserted without either OGX or Praxis serving traffic.
"""

import json
from dataclasses import dataclass
from typing import Any, TypedDict
from urllib.parse import quote

import structlog
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.config_map import ConfigMap
from ocp_resources.job import Job
from ocp_resources.pod import Pod
from ogx_client import OgxClient

from tests.ogx.constants import POSTGRESQL_PASSWORD, POSTGRESQL_USER
from tests.ogx.praxis.upgrade.constants import (
    API_BASELINE_CONFIG_MAP_KEY,
    API_BASELINE_CONFIG_MAP_NAME,
    COMPARED_FILE_FIELDS,
    COMPARED_VECTOR_STORE_FIELDS,
    MIGRATION_JOB_NAME_SUFFIX,
    MIGRATION_JOB_TIMEOUT,
    POSTGRES_CONTAINER_NAME,
    POSTGRES_PORT,
    PRAXIS_POSTGRES_DATABASE,
    PRAXIS_POSTGRES_SERVICE_NAME,
    SEED_FILE_PURPOSE,
    SEED_MARKER,
    SEED_RESPONSE_MAX_OUTPUT_TOKENS,
)
from utilities.exceptions import UnexpectedResourceCountError
from utilities.resources.ogx_server import OgxServer

LOGGER = structlog.get_logger(name=__name__)

# `psql` invocations are shell snippets so that credentials can be taken from the
# container environment instead of the argument list (Pod.execute logs the whole
# command). Callers pass SQL as a positional parameter, never by interpolation.
# Both PostgreSQL pods come from `get_postgres_deployment_template`, so the same
# snippet works against either of them.
_PSQL_SCRIPT: str = (
    'export PGPASSWORD="$POSTGRESQL_PASSWORD"; '
    'exec psql --host 127.0.0.1 --username "$POSTGRESQL_USER" --dbname "$1" '
    '--tuples-only --no-align --quiet --command "$2"'
)

# Number of mismatching rows quoted in an assertion message before truncating.
_MAX_REPORTED_ROWS: int = 10


class ApiBaseline(TypedDict):
    """Pre-upgrade Files and Vector Stores API responses, keyed by resource id."""

    files: dict[str, dict[str, str]]
    vector_stores: dict[str, dict[str, str]]


def _as_json_rows(select_statement: str) -> str:
    """Wrap a SELECT so psql emits one JSON object per row.

    JSON escapes newlines and separators inside values, so a line-per-row parse
    stays correct regardless of what the migrated payloads contain.

    Args:
        select_statement: A SELECT statement whose result columns become the
            keys of each emitted JSON object.

    Returns:
        A SELECT statement producing a single text column of JSON objects.
    """
    return f"SELECT row_to_json(migration_row)::text FROM ({select_statement}) AS migration_row"


def _parse_json_rows(psql_output: str) -> list[dict[str, Any]]:
    """Parse the one-JSON-object-per-line output of a `_as_json_rows` query.

    Args:
        psql_output: Raw stdout of the psql invocation.

    Returns:
        One dict per result row, in the order psql returned them.
    """
    return [json.loads(line) for line in psql_output.splitlines() if line.strip()]


def postgres_pod(client: DynamicClient, namespace: str, label_selector: str) -> Pod:
    """Return the single PostgreSQL pod matching `label_selector` in `namespace`.

    Args:
        client: Client with access to the test namespace.
        namespace: Namespace holding the PostgreSQL deployment.
        label_selector: Label selector identifying the instance; the OGX and
            Praxis deployments carry distinct `app` labels.

    Returns:
        The PostgreSQL pod.

    Raises:
        ResourceNotFoundError: If no PostgreSQL pod exists.
        UnexpectedResourceCountError: If more than one PostgreSQL pod exists.
    """
    pods = list(Pod.get(client=client, namespace=namespace, label_selector=label_selector))
    if not pods:
        raise ResourceNotFoundError(f"No pod found with label selector {label_selector} in namespace {namespace}")
    if len(pods) != 1:
        raise UnexpectedResourceCountError(
            f"Expected exactly 1 pod with label selector {label_selector} in namespace {namespace}, found {len(pods)}"
        )
    return pods[0]


def praxis_connection_string(namespace: str) -> str:
    """Build the DSN of the Praxis database deployed in `namespace`.

    The credentials are the ones `get_postgres_deployment_template` gives every
    instance it builds, so the DSN is derived rather than read back from the
    cluster. It is only ever written into a Secret, never logged or passed on a
    command line.

    Args:
        namespace: Namespace holding the Praxis PostgreSQL service.

    Returns:
        A `postgresql://` connection string for the Praxis database.
    """
    credentials = f"{quote(string=POSTGRESQL_USER, safe='')}:{quote(string=POSTGRESQL_PASSWORD, safe='')}"
    host = f"{PRAXIS_POSTGRES_SERVICE_NAME}.{namespace}.svc.cluster.local"
    return f"postgresql://{credentials}@{host}:{POSTGRES_PORT}/{PRAXIS_POSTGRES_DATABASE}"


def query_database(postgres_pod: Pod, database: str, select_statement: str) -> list[dict[str, Any]]:
    """Run a SELECT against a PostgreSQL instance by exec'ing psql in its pod.

    Args:
        postgres_pod: The PostgreSQL pod, source or target.
        database: Database name to connect to.
        select_statement: SELECT statement to run.

    Returns:
        One dict per result row.
    """
    output = postgres_pod.execute(
        command=["sh", "-c", _PSQL_SCRIPT, "sh", database, _as_json_rows(select_statement)],
        container=POSTGRES_CONTAINER_NAME,
    )
    return _parse_json_rows(psql_output=output)


def ids_present_in_table(
    postgres_pod: Pod,
    database: str,
    table: str,
    ids: list[str],
    id_column: str = "id",
) -> set[str]:
    """Return the subset of `ids` stored in `table`.

    Args:
        postgres_pod: The PostgreSQL pod to query.
        database: Database name to connect to.
        table: Unquoted table name in the `public` schema.
        ids: Resource ids to look for.
        id_column: Column holding the resource id. Every table uses `id` except
            the Praxis conversations table, which uses `conversation_id`.

    Returns:
        The ids found in the table.
    """
    id_literals = ", ".join(f"'{resource_id}'" for resource_id in ids)
    rows = query_database(
        postgres_pod=postgres_pod,
        database=database,
        select_statement=f"SELECT {id_column} AS resource_id FROM {table} WHERE {id_column} IN ({id_literals})",
    )
    return {str(row["resource_id"]) for row in rows}


def migration_target_secret_ref(ogx_server: OgxServer) -> dict[str, str] | None:
    """Return the Secret reference holding the Praxis target connection string.

    Args:
        ogx_server: The OGXServer whose spec is inspected.

    Returns:
        A dict with `name` and `key`, or None when the OGXServer does not opt
        into the migration Job (`spec.praxisMode.migrationJob` absent).
    """
    praxis_mode = ogx_server.instance.to_dict()["spec"].get("praxisMode") or {}
    migration_job = praxis_mode.get("migrationJob") or {}
    target_ref = migration_job.get("targetConnectionString")
    if not target_ref:
        return None
    return {"name": target_ref["name"], "key": target_ref["key"]}


def wait_for_migration_job_completion(client: DynamicClient, ogx_server: OgxServer) -> Job:
    """Wait until the operator-created Praxis migration Job reports Complete.

    Args:
        client: Client with access to the OGXServer namespace.
        ogx_server: The OGXServer that owns the migration Job.

    Returns:
        The completed migration Job.
    """
    job = Job(
        client=client,
        name=f"{ogx_server.name}{MIGRATION_JOB_NAME_SUFFIX}",
        namespace=ogx_server.namespace,
    )
    job.wait(timeout=MIGRATION_JOB_TIMEOUT)
    job.wait_for_condition(
        condition=Job.Condition.COMPLETE,
        status=Job.Condition.Status.TRUE,
        timeout=MIGRATION_JOB_TIMEOUT,
    )
    LOGGER.info(f"Praxis migration job {job.name} completed")
    return job


def seed_responses(ogx_client: OgxClient, model_id: str, count: int, conversation_id: str | None = None) -> list[str]:
    """Create stored responses so the OGX `openai_responses` table has rows to migrate.

    Responses are created through the OGX API rather than by direct INSERT so the
    stored `response_object` blobs are exactly what the migration's validation
    gate expects.

    Args:
        ogx_client: Client for the OGX server under test.
        model_id: Inference model used to produce the responses.
        count: Number of responses to create.
        conversation_id: When set, the first response is attached to this
            conversation so the conversation also carries continuity messages.

    Returns:
        The ids of the created responses.
    """
    response_ids: list[str] = []
    for index in range(count):
        extra_args: dict[str, Any] = {}
        if conversation_id and index == 0:
            extra_args["conversation"] = conversation_id
        response = ogx_client.responses.create(
            input=f"Reply with the single word '{SEED_MARKER}'. Sequence number {index}.",
            model=model_id,
            store=True,
            stream=False,
            max_output_tokens=SEED_RESPONSE_MAX_OUTPUT_TOKENS,
            **extra_args,
        )
        response_ids.append(response.id)
    LOGGER.info(f"Seeded {len(response_ids)} responses")
    return response_ids


def seed_conversations(ogx_client: OgxClient, count: int) -> list[str]:
    """Create conversations so the OGX `openai_conversations` table has rows to migrate.

    Args:
        ogx_client: Client for the OGX server under test.
        count: Number of conversations to create.

    Returns:
        The ids of the created conversations.
    """
    conversation_ids = [
        ogx_client.conversations.create(metadata={"source": SEED_MARKER, "sequence": str(index)}).id
        for index in range(count)
    ]
    LOGGER.info(f"Seeded {len(conversation_ids)} conversations")
    return conversation_ids


def seed_files(ogx_client: OgxClient, count: int) -> list[str]:
    """Upload files through the Files API so their ids can be re-read after the upgrade.

    The payload is generated in-process rather than read from the test corpus so
    that each file has a distinct, non-zero byte size; `bytes` is one of the
    fields compared across the upgrade.

    Args:
        ogx_client: Client for the OGX server under test.
        count: Number of files to upload.

    Returns:
        The ids of the uploaded files.
    """
    file_ids: list[str] = []
    for index in range(count):
        payload = f"{SEED_MARKER} file {index}\n{'x' * (index + 1) * 64}\n".encode()
        uploaded = ogx_client.files.create(
            file=(f"{SEED_MARKER}-{index}.txt", payload),
            purpose=SEED_FILE_PURPOSE,
        )
        file_ids.append(uploaded.id)
    LOGGER.info(f"Seeded {len(file_ids)} files")
    return file_ids


def capture_api_baseline(ogx_client: OgxClient, file_ids: list[str], vector_store_ids: list[str]) -> ApiBaseline:
    """Snapshot the Files and Vector Stores API responses for the given ids.

    Args:
        ogx_client: Client for the OGX server under test.
        file_ids: File ids to read through `GET /v1/files/{id}`.
        vector_store_ids: Vector store ids to read through
            `GET /v1/vector_stores/{id}`.

    Returns:
        The baseline, keyed by id, for comparison after the upgrade.
    """
    baseline: ApiBaseline = {
        "files": {
            file_id: _comparable_fields(
                payload=ogx_client.files.retrieve(file_id=file_id).to_dict(),
                fields=COMPARED_FILE_FIELDS,
            )
            for file_id in file_ids
        },
        "vector_stores": {
            vector_store_id: _comparable_fields(
                payload=ogx_client.vector_stores.retrieve(vector_store_id=vector_store_id).to_dict(),
                fields=COMPARED_VECTOR_STORE_FIELDS,
            )
            for vector_store_id in vector_store_ids
        },
    }
    LOGGER.info(f"Captured API baseline for {len(file_ids)} file(s) and {len(vector_store_ids)} vector store(s)")
    return baseline


def save_api_baseline_to_configmap(client: DynamicClient, namespace: str, baseline: ApiBaseline) -> ConfigMap:
    """Persist the pre-upgrade API baseline to a ConfigMap for the post-upgrade run.

    Args:
        client: Client with access to the test namespace.
        namespace: Namespace the baseline ConfigMap lives in.
        baseline: Snapshot to persist.

    Returns:
        The ConfigMap holding the baseline.
    """
    serialized = {API_BASELINE_CONFIG_MAP_KEY: json.dumps(baseline)}
    config_map = ConfigMap(client=client, name=API_BASELINE_CONFIG_MAP_NAME, namespace=namespace)
    if config_map.exists:
        resource_dict = config_map.instance.to_dict()
        resource_dict.setdefault("data", {}).update(serialized)
        config_map.update(resource_dict=resource_dict)
    else:
        config_map = ConfigMap(
            client=client,
            name=API_BASELINE_CONFIG_MAP_NAME,
            namespace=namespace,
            data=serialized,
        )
        config_map.deploy()
    LOGGER.info(f"Saved API baseline to ConfigMap {namespace}/{API_BASELINE_CONFIG_MAP_NAME}")
    return config_map


def load_api_baseline_from_configmap(client: DynamicClient, namespace: str) -> ApiBaseline:
    """Load the API baseline written by the pre-upgrade run.

    Args:
        client: Client with access to the test namespace.
        namespace: Namespace the baseline ConfigMap lives in.

    Returns:
        The persisted baseline.
    """
    config_map = ConfigMap(client=client, name=API_BASELINE_CONFIG_MAP_NAME, namespace=namespace)
    assert config_map.exists, (
        f"API baseline ConfigMap '{API_BASELINE_CONFIG_MAP_NAME}' not found in '{namespace}'. "
        "Ensure the pre-upgrade test ran successfully."
    )
    config_map_data = dict(config_map.instance.data or {})
    assert API_BASELINE_CONFIG_MAP_KEY in config_map_data, (
        f"API baseline ConfigMap '{API_BASELINE_CONFIG_MAP_NAME}' is missing the '{API_BASELINE_CONFIG_MAP_KEY}' key."
    )
    return json.loads(config_map_data[API_BASELINE_CONFIG_MAP_KEY])


def _comparable_fields(payload: dict[str, Any], fields: tuple[str, ...]) -> dict[str, str]:
    """Reduce an API response to the compared fields, stringified.

    Values are stringified so that a field surviving the upgrade with the same
    value but a different JSON numeric type does not register as a difference.

    Args:
        payload: Decoded API response body.
        fields: Field names to retain.

    Returns:
        The retained fields, as strings.
    """
    return {field: str(payload.get(field)) for field in fields}


def retrieve_file_fields(ogx_client: OgxClient, file_id: str) -> dict[str, str]:
    """Return the compared fields of `GET /v1/files/{id}`.

    Args:
        ogx_client: Client for the OGX server under test.
        file_id: File id to read.

    Returns:
        The compared fields, as strings.
    """
    return _comparable_fields(
        payload=ogx_client.files.retrieve(file_id=file_id).to_dict(),
        fields=COMPARED_FILE_FIELDS,
    )


def retrieve_vector_store_fields(ogx_client: OgxClient, vector_store_id: str) -> dict[str, str]:
    """Return the compared fields of `GET /v1/vector_stores/{id}`.

    Args:
        ogx_client: Client for the OGX server under test.
        vector_store_id: Vector store id to read.

    Returns:
        The compared fields, as strings.
    """
    return _comparable_fields(
        payload=ogx_client.vector_stores.retrieve(vector_store_id=vector_store_id).to_dict(),
        fields=COMPARED_VECTOR_STORE_FIELDS,
    )


def format_field_diff(resource: str, resource_id: str, before: dict[str, str], after: dict[str, str]) -> str:
    """Return an empty string when the fields match, otherwise a failure message.

    Args:
        resource: Human-readable resource kind, used in the message.
        resource_id: Id of the resource being compared.
        before: Pre-upgrade field values.
        after: Post-upgrade field values.

    Returns:
        A description of the differing fields, or "" when they are identical.
    """
    differences = [
        f"{field}: {before[field]!r} -> {after[field]!r}" for field in sorted(before) if before[field] != after[field]
    ]
    if not differences:
        return ""
    return f"{resource} '{resource_id}' changed across the upgrade on: " + "; ".join(differences)


@dataclass(frozen=True)
class TableParity:
    """Row-level comparison of one source table against its migrated counterpart."""

    table: str
    source_count: int
    target_count: int
    missing_ids: list[str]
    extra_ids: list[str]
    mismatched_ids: list[str]

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable form."""
        return {
            "table": self.table,
            "source_count": self.source_count,
            "target_count": self.target_count,
            "missing_in_target": self.missing_ids,
            "only_in_target": self.extra_ids,
            "field_mismatches": self.mismatched_ids,
        }


def compare_rows(
    table: str,
    source_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    source_key: str,
    target_key: str,
    compared_fields: tuple[str, ...],
) -> TableParity:
    """Compare source and target rows by primary key and by a set of fields.

    Values are stringified before comparison so that the JSON representations
    psql produces for the two schemas (for example `integer` versus `bigint`
    `created_at`) do not register as differences.

    Args:
        table: Logical table name, used in the result and in log messages.
        source_rows: Rows read from the OGX database.
        target_rows: Rows read from the Praxis database.
        source_key: Identity column name in `source_rows`.
        target_key: Identity column name in `target_rows`.
        compared_fields: Column names, present in both row sets, compared per id.

    Returns:
        The parity result for this table.
    """
    source = {str(row[source_key]): tuple(str(row[field]) for field in compared_fields) for row in source_rows}
    target = {str(row[target_key]): tuple(str(row[field]) for field in compared_fields) for row in target_rows}

    mismatched = sorted(row_id for row_id in source.keys() & target.keys() if source[row_id] != target[row_id])
    return TableParity(
        table=table,
        source_count=len(source),
        target_count=len(target),
        missing_ids=sorted(source.keys() - target.keys()),
        extra_ids=sorted(target.keys() - source.keys()),
        mismatched_ids=mismatched,
    )


def format_parity_failure(parity: TableParity, allow_extra_in_target: bool) -> str:
    """Return an empty string when `parity` is acceptable, otherwise a failure message.

    Args:
        parity: Comparison result to evaluate.
        allow_extra_in_target: When True, rows present only in the target are
            tolerated (the conversations phase synthesizes rows for
            `conversation_messages` orphans that have no source conversation).

    Returns:
        Human-readable description of the parity violations, or "" if there are none.
    """
    problems: list[str] = []
    if parity.missing_ids:
        problems.append(
            f"{len(parity.missing_ids)} row(s) missing from the Praxis table: {parity.missing_ids[:_MAX_REPORTED_ROWS]}"
        )
    if parity.mismatched_ids:
        problems.append(
            f"{len(parity.mismatched_ids)} row(s) migrated with differing field values: "
            f"{parity.mismatched_ids[:_MAX_REPORTED_ROWS]}"
        )
    if parity.extra_ids and not allow_extra_in_target:
        problems.append(
            f"{len(parity.extra_ids)} row(s) present only in the Praxis table: {parity.extra_ids[:_MAX_REPORTED_ROWS]}"
        )
    if not problems:
        return ""
    return (
        f"Table '{parity.table}' did not migrate with full parity "
        f"(source rows: {parity.source_count}, target rows: {parity.target_count}). " + "; ".join(problems)
    )
