import structlog
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.client import ApiException
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.job import Job
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from ocp_resources.utils.resource_constants import ResourceConstants
from timeout_sampler import TimeoutSampler

from tests.ogx.praxis.constants import (
    MIGRATION_FALLBACK_OWNER_SUBJECT,
    MIGRATION_FALLBACK_TENANT,
    OGX_CONNECTION_SECRET_KEY,
    OGX_POSTGRES_DATABASE,
    PRAXIS_CONNECTION_SECRET_KEY,
    PRAXIS_GATEWAY_POD_LABELS,
    SOURCE_RESPONSES_TABLE,
)
from tests.ogx.praxis.utils import query_database
from utilities.exceptions import MigrationJobFailedError
from utilities.resources.ogx_server import OgxServer

LOGGER = structlog.get_logger(name=__name__)

# Migration Job created by the OGX operator: `<ogxserver-name>-praxis-migration`.
_MIGRATION_JOB_NAME_SUFFIX: str = "-praxis-migration"
_MIGRATION_JOB_TIMEOUT: int = 600
_MIGRATION_JOB_POLL_INTERVAL: int = 2

# Tenancy column OGX adds to its tables once it runs multi-tenant. Enabling Praxis
# mode makes the operator rewrite the OGX config with `tenancy.mode: multi`, and OGX
# stamps the column on the next start, from `AuthorizedSqlStore.create_table`. The
# migration selects it unconditionally, so it cannot run before that restart has
# landed. The operator does not order the two: it creates the migration Job in the
# same reconcile that rewrites the config, and the Job's backoff limit expires in
# well under the time a pod needs to reschedule on a loaded cluster. The tests
# therefore enable Praxis mode and the migration in two steps, waiting for this
# column in between.
_SOURCE_TENANCY_COLUMN: str = "tenant_id"
_SOURCE_TENANCY_COLUMN_TIMEOUT: int = 600
_SOURCE_TENANCY_COLUMN_POLL_INTERVAL: int = 10


def run_migration(
    ogx_server: OgxServer,
    unprivileged_client: DynamicClient,
    ogx_connection_string_name: str,
    praxis_connection_string_name: str,
) -> None:
    """Enable Praxis mode and run the OGX-to-Praxis migration.

    Configures the OGXServer with the source and target connection strings,
    enables its migration Job, and waits for the operator-created Job to
    complete.

    Args:
        ogx_server: The OGXServer to configure and migrate.
        unprivileged_client: Client with access to the OGXServer namespace.
        ogx_connection_string_name: Name of the Secret containing the OGX
            database connection string.
        praxis_connection_string_name: Name of the Secret containing the
            Praxis database connection string.

    Raises:
        MigrationJobFailedError: If the migration Job reports Failed.
    """
    ResourceEditor(
        patches={
            ogx_server: {
                "spec": {
                    "storage": {
                        "sql": {
                            "type": "postgres",
                            "connectionString": {
                                "name": ogx_connection_string_name,
                                "key": OGX_CONNECTION_SECRET_KEY,
                            },
                        }
                    },
                    "praxisMode": {
                        "enabled": True,
                        "migrationJob": {
                            "enabled": True,
                            "targetConnectionString": {
                                "name": praxis_connection_string_name,
                                "key": PRAXIS_CONNECTION_SECRET_KEY,
                            },
                            "fallbackTenant": MIGRATION_FALLBACK_TENANT,
                            "fallbackOwnerSubject": MIGRATION_FALLBACK_OWNER_SUBJECT,
                        },
                    },
                }
            }
        }
    ).update()
    _wait_for_migration_job_completion(client=unprivileged_client, ogx_server=ogx_server)
    return ogx_server


def run_two_step_migration(
    ogx_server: OgxServer,
    unprivileged_client: DynamicClient,
    ogx_postgres_pod: Pod,
    ogx_connection_string_name: str,
    praxis_connection_string_name: str,
    namespace: str,
) -> None:
    """Enable Praxis mode, wait for tenancy support, and run the migration.

    Enables Praxis mode first and waits for OGX to add the tenancy column to
    the source table. It then enables the migration Job and waits for the
    operator-created Job to complete, avoiding the race between OGX startup
    and migration Job creation.

    Args:
        ogx_server: The OGXServer to configure and migrate.
        unprivileged_client: Client with access to the OGXServer namespace.
        ogx_postgres_pod: PostgreSQL pod containing the OGX source database.
        ogx_connection_string_name: Name of the Secret containing the OGX
            database connection string.
        praxis_connection_string_name: Name of the Secret containing the
            Praxis database connection string.
        namespace: Namespace containing the Praxis gateway.

    Raises:
        MigrationJobFailedError: If the migration Job reports Failed.
        TimeoutExpiredError: If the source table does not gain the tenancy
            column in time.
    """
    ResourceEditor(
        patches={
            ogx_server: {
                "spec": {
                    "storage": {
                        "sql": {
                            "type": "postgres",
                            "connectionString": {
                                "name": ogx_connection_string_name,
                                "key": OGX_CONNECTION_SECRET_KEY,
                            },
                        }
                    },
                    "praxisMode": {
                        "enabled": True,
                        "praxisSelector": {
                            "namespace": namespace,
                            "podSelector": {"matchLabels": PRAXIS_GATEWAY_POD_LABELS},
                        },
                    },
                }
            }
        }
    ).update()
    _wait_for_source_tenancy_column(
        postgres_pod=ogx_postgres_pod,
        database=OGX_POSTGRES_DATABASE,
        table=SOURCE_RESPONSES_TABLE,
    )

    ResourceEditor(
        patches={
            ogx_server: {
                "spec": {
                    "praxisMode": {
                        "migrationJob": {
                            "enabled": True,
                            "targetConnectionString": {
                                "name": praxis_connection_string_name,
                                "key": PRAXIS_CONNECTION_SECRET_KEY,
                            },
                            # Source rows seeded through the OGX API carry no
                            # tenancy, which the migration refuses to invent.
                            "fallbackTenant": MIGRATION_FALLBACK_TENANT,
                            "fallbackOwnerSubject": MIGRATION_FALLBACK_OWNER_SUBJECT,
                        },
                    },
                }
            }
        }
    ).update()
    _wait_for_migration_job_completion(client=unprivileged_client, ogx_server=ogx_server)
    return ogx_server


def _wait_for_migration_job_completion(client: DynamicClient, ogx_server: OgxServer) -> Job:
    """Wait until the operator-created Praxis migration Job reports Complete.

    A failed Job is reported as soon as it is seen, with the logs of its pods, rather
    than being left to time out. The logs are collected while the Job runs because the
    Job controller deletes a pod that exhausts the backoff limit, and it does so before
    the Job reports Failed -- by the time the failure is visible the pod is gone.

    Args:
        client: Client with access to the OGXServer namespace.
        ogx_server: The OGXServer that owns the migration Job.

    Returns:
        The completed migration Job.

    Raises:
        MigrationJobFailedError: If the Job reports Failed.
    """
    job = Job(
        client=client,
        name=f"{ogx_server.name}{_MIGRATION_JOB_NAME_SUFFIX}",
        namespace=ogx_server.namespace,
    )
    job.wait(timeout=_MIGRATION_JOB_TIMEOUT)

    captured: dict[str, str] = {}
    condition = ""
    for condition in TimeoutSampler(
        wait_timeout=_MIGRATION_JOB_TIMEOUT,
        sleep=_MIGRATION_JOB_POLL_INTERVAL,
        func=_capture_migration_job_logs,
        job=job,
        captured=captured,
    ):
        for condition in job.instance.status.conditions or []:
            if condition.status == ResourceConstants.Condition.Status.TRUE:
                if condition.type == ResourceConstants.Status.FAILED:
                    logs = (
                        "\n".join(f"--- pod {name} ---\n{log}" for name, log in captured.items())
                        or "no pod log was captured"
                    )
                    raise MigrationJobFailedError(f"Praxis migration job {job.name} failed:\n{logs}")

                if condition.type == ResourceConstants.Status.COMPLETE:
                    LOGGER.info(f"Praxis migration job {job.name} completed")
                    return job


def _capture_migration_job_logs(job: Job, captured: dict[str, str]) -> None:
    """Record the log of each pod the migration Job has created so far.

    Reads are repeated and overwrite earlier ones, so `captured` ends up holding
    each pod's final output. Pods that have gone away, or whose container has not
    produced a log yet, are skipped rather than failing the capture.

    Args:
        job: The migration Job whose pods are read.
        captured: Mapping of pod name to log, updated in place.
    """
    for pod in Pod.get(client=job.client, namespace=job.namespace, label_selector=f"job-name={job.name}"):
        try:
            log = pod.log()
        except ApiException, ResourceNotFoundError:
            continue
        if log:
            captured[pod.name] = log


def _wait_for_source_tenancy_column(postgres_pod: Pod, database: str, table: str) -> None:
    """Wait until an OGX source table carries the tenancy column.

    OGX adds `tenant_id` only when it starts with a multi-tenant config, which is
    what enabling Praxis mode makes the operator write. The migration selects the
    column unconditionally, so this is the precondition that makes it runnable.

    Args:
        postgres_pod: The PostgreSQL pod holding the source database.
        database: Source database name.
        table: Unquoted source table name in the `public` schema.

    Raises:
        TimeoutExpiredError: If the column does not appear in time.
    """
    for rows in TimeoutSampler(
        wait_timeout=_SOURCE_TENANCY_COLUMN_TIMEOUT,
        sleep=_SOURCE_TENANCY_COLUMN_POLL_INTERVAL,
        func=query_database,
        postgres_pod=postgres_pod,
        database=database,
        select_statement=(
            "SELECT column_name FROM information_schema.columns "
            f"WHERE table_schema = 'public' AND table_name = '{table}' "
            f"AND column_name = '{_SOURCE_TENANCY_COLUMN}'"
        ),
    ):
        if rows:
            LOGGER.info(f"Source table {table} carries {_SOURCE_TENANCY_COLUMN}; migration can read it")
            break
