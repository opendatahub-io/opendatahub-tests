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
    PRAXIS_CONNECTION_SECRET_KEY,
)
from utilities.exceptions import MigrationJobFailedError
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
