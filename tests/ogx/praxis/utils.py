"""Helpers shared by the OGX -> Praxis migration tests.

The Gateway helpers resolve what actually serves an API path from the cluster
itself: a path is matched against the HTTPRoutes that declare it. Nothing about
Praxis naming is hardcoded, so a missing or ambiguous route cannot pass unnoticed.
"""

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, TypedDict
from urllib.parse import quote

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.client import ApiException
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.config_map import ConfigMap
from ocp_resources.job import Job
from ocp_resources.pod import Pod
from ocp_resources.route import Route
from ocp_resources.service import Service
from ogx_client import APIStatusError, OgxClient

from tests.ogx.constants import OGX_POSTGRES_SERVICE_NAME, POSTGRESQL_PASSWORD, POSTGRESQL_USER
from tests.ogx.praxis.constants import (
    API_BASELINE_CONFIG_MAP_NAME,
    COMPARED_CONVERSATION_FIELDS,
    COMPARED_FILE_FIELDS,
    COMPARED_RESPONSE_FIELDS,
    COMPARED_VECTOR_STORE_FIELDS,
    MAX_LISTED_RESOURCES,
    MIGRATION_JOB_NAME_SUFFIX,
    MIGRATION_JOB_TIMEOUT,
    OGX_POSTGRES_DATABASE,
    POSTGRES_CONTAINER_NAME,
    POSTGRES_PORT,
    PRAXIS_POSTGRES_DATABASE,
    PRAXIS_POSTGRES_SERVICE_NAME,
    SEED_FILE_PURPOSE,
    SEED_MARKER,
    SEED_RESPONSE_MAX_OUTPUT_TOKENS,
)
from utilities.exceptions import UnexpectedResourceCountError
from utilities.resources.http_route import HTTPRoute
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


class FileSearchCitationBaseline(TypedDict):
    """Pre-upgrade inputs for the file_search citation check.

    `file_ids` are the files attached to `vector_store_id` before the upgrade;
    every citation returned afterwards must reference one of them.
    """

    vector_store_id: str
    file_ids: list[str]


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


def rule_matches_path(rule: dict[str, Any], path: str) -> bool:
    """Return whether an HTTPRoute rule declares an exact match on `path`.

    Args:
        rule: A single entry of `spec.rules`.
        path: Absolute request path, as declared by a route rule match.

    Returns:
        True if any of the rule matches declares exactly `path`.
    """
    return any((match.get("path") or {}).get("value") == path for match in rule.get("matches") or [])


def http_routes_matching_path(client: DynamicClient, path: str, namespace: str | None = None) -> list[HTTPRoute]:
    """Return every HTTPRoute in a namespace that explicitly declares `path`.

    Catch-all routes (for example a `PathPrefix: /` rule) are deliberately not
    treated as matches; the contract under test is about the route that owns the
    exact path. When no namespace is provided, routes are searched cluster-wide.

    Args:
        client: Client with cluster-wide read access.
        path: Absolute request path, as declared by a route rule match.
        namespace: Namespace to search, or None to search every namespace.

    Returns:
        The routes declaring an exact match on `path`.
    """
    return [
        http_route
        for http_route in HTTPRoute.get(client=client, namespace=namespace)
        if any(
            rule_matches_path(rule=rule, path=path) for rule in http_route.instance.to_dict()["spec"].get("rules") or []
        )
    ]


def backend_services(client: DynamicClient, http_route: HTTPRoute, path: str) -> list[Service]:
    """Resolve the Service backends an HTTPRoute forwards `path` to.

    Only the rules declaring an exact match on `path` are resolved, so a route
    carrying rules for several paths cannot attribute another path's backends
    to `path`.

    Args:
        client: Client with cluster-wide read access.
        http_route: Route whose `backendRefs` are resolved.
        path: Absolute request path the backends must serve.

    Returns:
        One Service per distinct namespace/name backend reference.
    """
    resolved: dict[tuple[str, str], Service] = {}
    for rule in http_route.instance.to_dict()["spec"].get("rules") or []:
        if not rule_matches_path(rule=rule, path=path):
            continue
        for backend_ref in rule.get("backendRefs") or []:
            if (backend_ref.get("kind") or "Service") != "Service":
                continue
            namespace: str = str(backend_ref.get("namespace") or http_route.namespace)
            name: str = str(backend_ref["name"])
            resolved.setdefault(
                (namespace, name),
                Service(client=client, name=name, namespace=namespace),
            )
    return list(resolved.values())


def pods_for_service(client: DynamicClient, service: Service) -> list[Pod]:
    """Return the pods selected by a Service.

    Args:
        client: Client with read access to the Service namespace.
        service: Service whose `spec.selector` is used.

    Returns:
        The matching pods; empty when the Service has no selector.
    """
    selector = service.instance.to_dict()["spec"].get("selector") or {}
    if not selector:
        return []
    label_selector = ",".join(f"{key}={value}" for key, value in sorted(selector.items()))
    return list(Pod.get(client=client, namespace=service.namespace, label_selector=label_selector))


def serving_pods_for_path(client: DynamicClient, http_route: HTTPRoute, path: str) -> list[Pod]:
    """Return the pods serving `path`, through the route's existing backend Services.

    Args:
        client: Client with cluster-wide read access.
        http_route: Route whose serving workload is resolved.
        path: Absolute request path whose serving workload is resolved.

    Returns:
        The pods selected by the backend Services of the rules matching `path`.
    """
    return [
        pod
        for service in backend_services(client=client, http_route=http_route, path=path)
        if service.exists
        for pod in pods_for_service(client=client, service=service)
    ]


def pod_logs(pod: Pod) -> str:
    """Return the concatenated logs of every container in a pod.

    Args:
        pod: Pod to read logs from.

    Returns:
        All container logs joined by newlines.
    """
    return "\n".join(
        pod.log(container=container["name"]) for container in pod.instance.to_dict()["spec"].get("containers") or []
    )


def route_url(route: Route, path: str) -> str:
    """Build the external URL of a path exposed by an OpenShift Route.

    Args:
        route: Route providing the host and TLS configuration.
        path: Absolute request path.

    Returns:
        The externally reachable URL.
    """
    route_spec = route.instance.to_dict()["spec"]
    scheme = "https" if route_spec.get("tls") else "http"
    return f"{scheme}://{route_spec['host']}{path}"


def gateway_base_url(http_route: HTTPRoute) -> str:
    """Return the external base URL of the Gateway exposing an HTTPRoute.

    Args:
        http_route: Route whose `spec.hostnames` provides the external hostname.

    Returns:
        The `https://<hostname>` base URL, without a trailing slash.

    Raises:
        ValueError: If the route exposes no hostname, so it is not reachable
            from outside the cluster.
    """
    hostnames = http_route.instance.to_dict()["spec"].get("hostnames") or []
    if not hostnames:
        raise ValueError(
            f"HTTPRoute {http_route.namespace}/{http_route.name} exposes no hostname, "
            "so it is not reachable from outside the cluster"
        )
    return f"https://{hostnames[0]}"


def praxis_http_route(client: DynamicClient, path: str) -> HTTPRoute:
    """Return the single HTTPRoute publishing an API path at the public boundary.

    The owning route is resolved from the cluster so nothing about Praxis naming is
    hardcoded. A path no route declares means Praxis does not front that API here.

    Args:
        client: Client with cluster-wide read access.
        path: Absolute request path, for example `/v1/files`.

    Returns:
        The route declaring an exact match on `path`.

    Raises:
        UnexpectedResourceCountError: If more than one HTTPRoute declares `path`, making
            the public boundary ambiguous.
    """
    http_routes = http_routes_matching_path(client=client, path=path)
    if not http_routes:
        pytest.skip(
            f"No HTTPRoute declares {path}; Praxis is not the public entrypoint on this cluster. "
            f"Deploy Praxis with a Gateway API route for {path} to run this test."
        )
    if len(http_routes) > 1:
        raise UnexpectedResourceCountError(
            f"Expected exactly one HTTPRoute to own {path}, found "
            f"{[f'{route.namespace}/{route.name}' for route in http_routes]}; the public boundary is ambiguous"
        )
    return http_routes[0]


def praxis_api_url(client: DynamicClient, path: str) -> str:
    """Return the public URL of an OpenAI-compatible API path served by Praxis.

    Args:
        client: Client with cluster-wide read access.
        path: Absolute request path, for example `/v1/files`.

    Returns:
        The externally reachable URL of `path`.
    """
    return f"{gateway_base_url(http_route=praxis_http_route(client=client, path=path))}{path}"


def praxis_gateway_base_url(client: DynamicClient, paths: Iterable[str]) -> str:
    """Return the single external base URL serving every given API path.

    Each path is resolved through the HTTPRoute that owns it, so the base URL is the
    boundary a client outside the cluster actually addresses. The paths are required to
    agree: a split external boundary is a finding in itself, not something to pick a
    winner from. A path no route declares skips the test, as `praxis_http_route` does.

    Args:
        client: Client with cluster-wide read access.
        paths: Absolute request paths that must share one public hostname.

    Returns:
        The `https://<hostname>` base URL shared by all `paths`.

    Raises:
        UnexpectedResourceCountError: If the paths resolve to more than one base URL.
    """
    base_urls = {path: gateway_base_url(http_route=praxis_http_route(client=client, path=path)) for path in paths}
    if len(set(base_urls.values())) > 1:
        raise UnexpectedResourceCountError(
            f"Expected one external base URL for all API paths, found {base_urls}; "
            "the public boundary is split across hostnames"
        )
    return next(iter(base_urls.values()))


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


def postgres_connection_string(namespace: str, service_name: str, database: str) -> str:
    """Build a PostgreSQL DSN for a database deployed in `namespace`.

    The credentials are the ones `get_postgres_deployment_template` gives every
    instance it builds, so the DSN is derived rather than read back from the
    cluster. It is only ever written into a Secret, never logged or passed on a
    command line.

    Args:
        namespace: Namespace holding the PostgreSQL service.
        service_name: Name of the Service fronting PostgreSQL.
        database: Name of the PostgreSQL database.

    Returns:
        A `postgresql://` connection string for the requested database.
    """
    credentials = f"{quote(string=POSTGRESQL_USER, safe='')}:{quote(string=POSTGRESQL_PASSWORD, safe='')}"
    host = f"{service_name}.{namespace}.svc.cluster.local"
    return f"postgresql://{credentials}@{host}:{POSTGRES_PORT}/{database}"


def ogx_connection_string(namespace: str) -> str:
    """Build the DSN of the source OGX database deployed in `namespace`."""
    return postgres_connection_string(
        namespace=namespace,
        service_name=OGX_POSTGRES_SERVICE_NAME,
        database=OGX_POSTGRES_DATABASE,
    )


def praxis_connection_string(namespace: str) -> str:
    """Build the DSN of the target Praxis database deployed in `namespace`."""
    return postgres_connection_string(
        namespace=namespace,
        service_name=PRAXIS_POSTGRES_SERVICE_NAME,
        database=PRAXIS_POSTGRES_DATABASE,
    )


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


def save_baseline_section(client: DynamicClient, namespace: str, section: str, payload: Any) -> ConfigMap:
    """Persist one pre-upgrade baseline section to the shared baseline ConfigMap.

    Every pre-upgrade test in a namespace writes its own `section`, stored as a
    JSON document under its own ConfigMap data key. Writing is a merge patch of
    that single key, so tests sharing the namespace never overwrite each other's
    section and no read-modify-write of the whole ConfigMap is needed.

    Args:
        client: Client with access to the test namespace.
        namespace: Namespace the baseline ConfigMap lives in.
        section: ConfigMap data key identifying the writing test's section.
        payload: JSON-serializable snapshot to persist under `section`.

    Returns:
        The ConfigMap holding the baseline.
    """
    serialized = {section: json.dumps(payload)}
    config_map = ConfigMap(client=client, name=API_BASELINE_CONFIG_MAP_NAME, namespace=namespace)
    if config_map.exists:
        config_map.update(resource_dict={"data": serialized})
    else:
        config_map = ConfigMap(
            client=client,
            name=API_BASELINE_CONFIG_MAP_NAME,
            namespace=namespace,
            data=serialized,
        )
        config_map.deploy()
    LOGGER.info(f"Saved baseline section '{section}' to ConfigMap {namespace}/{API_BASELINE_CONFIG_MAP_NAME}")
    return config_map


def load_baseline_section(client: DynamicClient, namespace: str, section: str) -> Any:
    """Load one baseline section written by the pre-upgrade run.

    Args:
        client: Client with access to the test namespace.
        namespace: Namespace the baseline ConfigMap lives in.
        section: ConfigMap data key the section was written under.

    Returns:
        The deserialized section payload.
    """
    config_map = ConfigMap(client=client, name=API_BASELINE_CONFIG_MAP_NAME, namespace=namespace)
    assert config_map.exists, (
        f"Baseline ConfigMap '{API_BASELINE_CONFIG_MAP_NAME}' not found in '{namespace}'. "
        "Ensure the pre-upgrade test ran successfully."
    )
    config_map_data = dict(config_map.instance.data or {})
    assert section in config_map_data, (
        f"Baseline ConfigMap '{API_BASELINE_CONFIG_MAP_NAME}' in '{namespace}' is missing the '{section}' key; "
        f"it carries {sorted(config_map_data)}. Ensure the pre-upgrade test writing that section ran successfully."
    )
    return json.loads(config_map_data[section])


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
        problems.append(f"{len(parity.missing_ids)} row(s) missing from the Praxis table: {parity.missing_ids}")
    if parity.mismatched_ids:
        problems.append(
            f"{len(parity.mismatched_ids)} row(s) migrated with differing field values: {parity.mismatched_ids}"
        )
    if parity.extra_ids and not allow_extra_in_target:
        problems.append(f"{len(parity.extra_ids)} row(s) present only in the Praxis table: {parity.extra_ids}")
    if not problems:
        return ""
    return (
        f"Table '{parity.table}' did not migrate with full parity "
        f"(source rows: {parity.source_count}, target rows: {parity.target_count}). " + "; ".join(problems)
    )


def retrieve_response_fields(ogx_client: OgxClient, response_id: str) -> dict[str, str]:
    """Return the compared fields of `GET /v1/responses/{id}`.

    Args:
        ogx_client: Client for the server under test.
        response_id: Response id to read.

    Returns:
        The compared fields, as strings.
    """
    return _comparable_fields(
        payload=ogx_client.responses.retrieve(response_id=response_id).to_dict(),
        fields=COMPARED_RESPONSE_FIELDS,
    )


def retrieve_conversation_fields(ogx_client: OgxClient, conversation_id: str) -> dict[str, str]:
    """Return the compared fields of `GET /v1/conversations/{id}`.

    Args:
        ogx_client: Client for the server under test.
        conversation_id: Conversation id to read.

    Returns:
        The compared fields, as strings.
    """
    return _comparable_fields(
        payload=ogx_client.conversations.retrieve(conversation_id=conversation_id).to_dict(),
        fields=COMPARED_CONVERSATION_FIELDS,
    )


def bounded_count(items: Iterable[Any], max_items: int = MAX_LISTED_RESOURCES) -> int:
    """Count an auto-paginating listing, refusing to walk an unbounded one.

    Args:
        items: Listing returned by one of the client's `list()` methods.
        max_items: Largest number of items the caller is willing to walk.

    Returns:
        The number of listed items.

    Raises:
        UnexpectedResourceCountError: If the listing holds more than `max_items`.
    """
    count = 0
    for _ in items:
        count += 1
        if count > max_items:
            raise UnexpectedResourceCountError(f"Listing holds more than the {max_items} items the inventory walks")
    return count


class StateInventory(TypedDict):
    """Snapshot of the state a disruptive operation must preserve.

    `counts` holds one total per resource kind. `sampled` holds the compared
    fields of the individual resources named by the caller, keyed by kind and
    then by id. `missing` names the sampled resources that did not resolve.
    """

    counts: dict[str, int]
    sampled: dict[str, dict[str, dict[str, str]]]
    missing: list[str]


class RollbackBaseline(TypedDict):
    """Pre-rollback state inventory, together with the URL it was captured through.

    `external_url` is recorded so the post-rollback run can show it addressed the
    same hostname, rather than one the rollback moved the endpoint to.
    """

    inventory: StateInventory
    external_url: str


def capture_state_inventory(ogx_client: OgxClient, sampled_ids: dict[str, list[str]]) -> StateInventory:
    """Count the stored resources and read back the sampled ones.

    The Conversations API exposes no listing endpoint, so its count is the number
    of sampled conversations that resolved rather than a server-wide total.

    Args:
        ogx_client: Client for the server under test.
        sampled_ids: Resource ids to read back, keyed by `files`,
            `vector_stores`, `responses` and `conversations`.

    Returns:
        The inventory, ready to be compared against another capture.
    """
    retrievers: dict[str, Callable[..., dict[str, str]]] = {
        "files": lambda resource_id: retrieve_file_fields(ogx_client=ogx_client, file_id=resource_id),
        "vector_stores": lambda resource_id: retrieve_vector_store_fields(
            ogx_client=ogx_client, vector_store_id=resource_id
        ),
        "responses": lambda resource_id: retrieve_response_fields(ogx_client=ogx_client, response_id=resource_id),
        "conversations": lambda resource_id: retrieve_conversation_fields(
            ogx_client=ogx_client, conversation_id=resource_id
        ),
    }

    sampled: dict[str, dict[str, dict[str, str]]] = {kind: {} for kind in retrievers}
    missing: list[str] = []
    for kind, resource_ids in sampled_ids.items():
        for resource_id in resource_ids:
            try:
                sampled[kind][resource_id] = retrievers[kind](resource_id=resource_id)
            except APIStatusError as error:
                missing.append(f"{kind}/{resource_id} returned HTTP {error.status_code}")

    counts = {
        "files": bounded_count(items=ogx_client.files.list()),
        "vector_stores": bounded_count(items=ogx_client.vector_stores.list()),
        "responses": bounded_count(items=ogx_client.responses.list()),
        "conversations": len(sampled["conversations"]),
    }
    LOGGER.info(f"Captured state inventory {counts}")
    return StateInventory(counts=counts, sampled=sampled, missing=missing)


def sampled_resource_ids(inventory: StateInventory) -> dict[str, list[str]]:
    """Return the ids an inventory sampled, keyed by resource kind.

    Lets a later capture re-read exactly the resources an earlier one recorded,
    without the ids having to be carried separately.

    Args:
        inventory: A previously captured inventory.

    Returns:
        The sampled ids, in the shape `capture_state_inventory` expects.
    """
    return {kind: list(sampled) for kind, sampled in inventory["sampled"].items()}


def pods_logging_marker(pods: list[Pod], marker: str) -> list[str]:
    """Return the names of the pods whose logs contain `marker`.

    Used to correlate a single request against the workload that served it. A
    pod whose logs cannot be read (for example because it has already been
    replaced) is reported as not containing the marker.

    Args:
        pods: Pods whose logs are searched.
        marker: Literal string to look for, typically a request or response id.

    Returns:
        The names of the matching pods.
    """
    matching: list[str] = []
    for pod in pods:
        try:
            if marker in pod_logs(pod=pod):
                matching.append(str(pod.name))
        except ApiException as error:
            LOGGER.warning(f"Could not read logs of pod {pod.namespace}/{pod.name}: {error}")
    return matching
