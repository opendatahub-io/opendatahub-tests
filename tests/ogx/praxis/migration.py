import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.job import Job
from ocp_resources.resource import ResourceEditor
from ocp_resources.utils.resource_constants import ResourceConstants
from timeout_sampler import TimeoutSampler

from tests.ogx.praxis.constants import (
    MIGRATION_FALLBACK_OWNER_SUBJECT,
    MIGRATION_FALLBACK_TENANT,
    OGX_CONNECTION_SECRET_KEY,
    PRAXIS_CONNECTION_SECRET_KEY,
    PRAXIS_GATEWAY_POD_LABELS,
)
from utilities.resources.ogx_server import OgxServer

LOGGER = structlog.get_logger(name=__name__)

# Migration Job created by the OGX operator: `<ogxserver-name>-praxis-migration`.
_MIGRATION_JOB_NAME_SUFFIX: str = "-praxis-migration"
_MIGRATION_JOB_TIMEOUT: int = 600
_MIGRATION_JOB_POLL_INTERVAL: int = 2
_PRAXIS_REACHABLE_CONDITION: str = "PraxisReachable"
_TLS_CONFIGURED_CONDITION: str = "TLSConfigured"
_DEPLOYMENT_ROLLOUT_TIMEOUT: int = 240


def wait_for_deployment_rollout(
    deployment: Deployment,
    previous_generation: int,
) -> None:
    """Wait for a new Deployment generation to replace all previous replicas.

    The migration Job can complete before the operator finishes rolling out the
    Praxis-mode OGX configuration. Waiting only for ready and updated replicas is
    also insufficient because an old surge replica can remain ready and receive
    traffic. This wait therefore requires a newer observed generation and full
    convergence of the total, updated, available, and ready replica counts.

    Args:
        deployment: Deployment updated as part of the migration.
        previous_generation: Deployment generation observed before migration.
    """
    for instance in TimeoutSampler(
        wait_timeout=_DEPLOYMENT_ROLLOUT_TIMEOUT,
        sleep=_MIGRATION_JOB_POLL_INTERVAL,
        func=lambda: deployment.instance,
    ):
        generation = int(instance.metadata.generation or 0)
        desired_replicas = int(instance.spec.replicas or 0)
        status = instance.status

        if (
            generation > previous_generation
            and int(status.observedGeneration or 0) >= generation
            and int(status.replicas or 0) == desired_replicas
            and int(status.updatedReplicas or 0) == desired_replicas
            and int(status.availableReplicas or 0) == desired_replicas
            and int(status.readyReplicas or 0) == desired_replicas
        ):
            return


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
                        "praxisSelector": {
                            "namespace": ogx_server.namespace,
                            "podSelector": {"matchLabels": PRAXIS_GATEWAY_POD_LABELS},
                        },
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
    """Wait until the operator-created Praxis migration Job finishes.

    Args:
        client: Client with access to the OGXServer namespace.
        ogx_server: The OGXServer that owns the migration Job.

    Returns:
        The completed migration Job.

    Raises:
        RuntimeError: If the migration Job reports Failed.
    """
    job = Job(
        client=client,
        name=f"{ogx_server.name}{_MIGRATION_JOB_NAME_SUFFIX}",
        namespace=ogx_server.namespace,
    )
    job.wait(timeout=_MIGRATION_JOB_TIMEOUT)

    for instance in TimeoutSampler(
        wait_timeout=_MIGRATION_JOB_TIMEOUT,
        sleep=_MIGRATION_JOB_POLL_INTERVAL,
        func=lambda: job.instance,
    ):
        if not instance.status or not instance.status.conditions:
            continue

        for condition in instance.status.conditions:
            if condition.status != ResourceConstants.Condition.Status.TRUE:
                continue

            if condition.type == ResourceConstants.Status.FAILED:
                raise RuntimeError(
                    f"Praxis migration job {job.name} failed: "
                    f"{condition.reason or 'unknown reason'}: {condition.message or 'no message'}"
                )

            if condition.type == ResourceConstants.Status.COMPLETE:
                LOGGER.info(f"Praxis migration job {job.name} completed")
                return job
