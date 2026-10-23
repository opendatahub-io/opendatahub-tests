import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.job import Job
from ocp_resources.resource import ResourceEditor

from tests.ogx.praxis.constants import (
    MIGRATION_FALLBACK_OWNER_SUBJECT,
    MIGRATION_FALLBACK_TENANT,
    OGX_CONNECTION_SECRET_KEY,
    PRAXIS_CONNECTION_SECRET_KEY,
)
from utilities.resources.ogx_server import OgxServer

LOGGER = structlog.get_logger(name=__name__)

# Migration Job created by the OGX operator: `<ogxserver-name>-praxis-migration`.
_MIGRATION_JOB_NAME_SUFFIX: str = "-praxis-migration"
_MIGRATION_JOB_TIMEOUT: int = 600
_MIGRATION_JOB_POLL_INTERVAL: int = 2


def run_migration(
    ogx_server: OgxServer,
    unprivileged_client: DynamicClient,
    ogx_connection_string_name: str,
    praxis_connection_string_name: str,
) -> OgxServer:
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


def _wait_for_migration_job_completion(client: DynamicClient, ogx_server: OgxServer) -> Job:
    """Wait until the operator-created Praxis migration Job reports Complete.

    Args:
        client: Client with access to the OGXServer namespace.
        ogx_server: The OGXServer that owns the migration Job.

    Returns:
        The completed migration Job.
    """
    job = Job(
        client=client,
        name=f"{ogx_server.name}{_MIGRATION_JOB_NAME_SUFFIX}",
        namespace=ogx_server.namespace,
    )
    job.wait(timeout=_MIGRATION_JOB_TIMEOUT)
    job.wait_for_condition(
        condition=Job.Condition.COMPLETE,
        status=Job.Condition.Status.TRUE,
        timeout=_MIGRATION_JOB_TIMEOUT,
    )
    LOGGER.info(f"Praxis migration job {job.name} completed")
    return job
