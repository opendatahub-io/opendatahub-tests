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
from typing import Any
from urllib.parse import quote

import structlog
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.job import Job
from ocp_resources.pod import Pod
from ogx_client import OgxClient

from tests.ogx.constants import POSTGRESQL_PASSWORD, POSTGRESQL_USER
from tests.ogx.praxis.constants import (
    MIGRATION_JOB_NAME_SUFFIX,
    MIGRATION_JOB_TIMEOUT,
    POSTGRES_CONTAINER_NAME,
    POSTGRES_PORT,
    PRAXIS_POSTGRES_DATABASE,
    PRAXIS_POSTGRES_SERVICE_NAME,
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
