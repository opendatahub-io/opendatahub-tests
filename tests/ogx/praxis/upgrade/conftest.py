from collections.abc import Callable, Generator

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from ocp_resources.secret import Secret
from ocp_resources.service import Service

from tests.ogx.praxis.upgrade.constants import (
    OGX_CONNECTION_SECRET_KEY,
    OGX_CONNECTION_SECRET_NAME,
    OGX_POSTGRES_POD_LABEL_SELECTOR,
    POSTGRES_PORT,
    PRAXIS_CONNECTION_SECRET_KEY,
    PRAXIS_CONNECTION_SECRET_NAME,
    PRAXIS_POSTGRES_APP_LABEL,
    PRAXIS_POSTGRES_DATABASE,
    PRAXIS_POSTGRES_DEPLOYMENT_NAME,
    PRAXIS_POSTGRES_POD_LABEL_SELECTOR,
    PRAXIS_POSTGRES_SERVICE_NAME,
)
from tests.ogx.praxis.upgrade.utils import (
    ogx_connection_string,
    postgres_pod,
    praxis_connection_string,
    wait_for_migration_job_completion,
)
from tests.ogx.utils import get_postgres_deployment_template
from utilities.resources.ogx_server import OgxServer


@pytest.fixture(scope="class")
def ogx_postgres_pod(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    postgres_deployment: Deployment,
    ogx_server: OgxServer,
) -> Pod:
    """The PostgreSQL pod holding the OGX (source) database."""
    return postgres_pod(
        client=unprivileged_client,
        namespace=unprivileged_model_namespace.name,
        label_selector=OGX_POSTGRES_POD_LABEL_SELECTOR,
    )


@pytest.fixture(scope="class")
def praxis_postgres_deployment(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    ogx_server_secret: Secret,
    teardown_resources: bool,
) -> Generator[Deployment]:
    """Deploy the Praxis (target) PostgreSQL instance.

    Built from the same template as the OGX instance, so it shares its
    credentials, but with its own app label and database.
    """
    deployment = Deployment(
        client=unprivileged_client,
        namespace=unprivileged_model_namespace.name,
        name=PRAXIS_POSTGRES_DEPLOYMENT_NAME,
        min_ready_seconds=5,
        replicas=1,
        selector={"matchLabels": {"app": PRAXIS_POSTGRES_APP_LABEL}},
        strategy={"type": "Recreate"},
        template=get_postgres_deployment_template(
            app_label=PRAXIS_POSTGRES_APP_LABEL, database=PRAXIS_POSTGRES_DATABASE
        ),
        teardown=teardown_resources,
    )
    with deployment:
        deployment.wait_for_replicas(deployed=True, timeout=240)
        yield deployment


@pytest.fixture(scope="class")
def praxis_postgres_service(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    praxis_postgres_deployment: Deployment,
    teardown_resources: bool,
) -> Generator[Service]:
    """Service fronting the Praxis PostgreSQL instance, as addressed by the target DSN."""
    service = Service(
        client=unprivileged_client,
        namespace=unprivileged_model_namespace.name,
        name=PRAXIS_POSTGRES_SERVICE_NAME,
        ports=[{"port": POSTGRES_PORT, "targetPort": POSTGRES_PORT}],
        selector={"app": PRAXIS_POSTGRES_APP_LABEL},
        wait_for_resource=True,
        teardown=teardown_resources,
    )
    with service:
        yield service


@pytest.fixture(scope="class")
def ogx_connection_string_secret(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    postgres_service: Service,
    teardown_resources: bool,
) -> Generator[Secret]:
    """Secret holding the source OGX PostgreSQL DSN."""
    with Secret(
        client=unprivileged_client,
        namespace=unprivileged_model_namespace.name,
        name=OGX_CONNECTION_SECRET_NAME,
        labels={"ogx.io/watch": "true"},
        type="Opaque",
        string_data={OGX_CONNECTION_SECRET_KEY: ogx_connection_string(namespace=unprivileged_model_namespace.name)},
        teardown=teardown_resources,
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def praxis_connection_string_secret(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    praxis_postgres_service: Service,
    teardown_resources: bool,
) -> Generator[Secret]:
    """Secret holding the Praxis DSN for `migrationJob.targetConnectionString`.

    The test references this Secret when it enables the migration Job.
    """
    secret = Secret(
        client=unprivileged_client,
        namespace=unprivileged_model_namespace.name,
        name=PRAXIS_CONNECTION_SECRET_NAME,
        labels={"ogx.io/watch": "true"},
        type="Opaque",
        string_data={
            PRAXIS_CONNECTION_SECRET_KEY: praxis_connection_string(namespace=unprivileged_model_namespace.name)
        },
        teardown=teardown_resources,
    )
    with secret:
        yield secret


@pytest.fixture(scope="class")
def praxis_migration_runner(
    unprivileged_client: DynamicClient,
    ogx_server: OgxServer,
    ogx_connection_string_secret: Secret,
    praxis_connection_string_secret: Secret,
) -> Callable[[], OgxServer]:
    """Return a runner that enables Praxis mode and waits for database migration."""

    def _run_migration() -> OgxServer:
        ResourceEditor(
            patches={
                ogx_server: {
                    "spec": {
                        "storage": {
                            "sql": {
                                "type": "postgres",
                                "connectionString": {
                                    "name": ogx_connection_string_secret.name,
                                    "key": OGX_CONNECTION_SECRET_KEY,
                                },
                            }
                        },
                        "praxisMode": {
                            "enabled": True,
                            "migrationJob": {
                                "enabled": True,
                                "targetConnectionString": {
                                    "name": praxis_connection_string_secret.name,
                                    "key": PRAXIS_CONNECTION_SECRET_KEY,
                                },
                            },
                        },
                    }
                }
            }
        ).update()
        wait_for_migration_job_completion(client=unprivileged_client, ogx_server=ogx_server)
        return ogx_server

    return _run_migration


@pytest.fixture(scope="class")
def praxis_postgres_pod(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    praxis_postgres_deployment: Deployment,
) -> Pod:
    """The PostgreSQL pod holding the Praxis (target) database."""
    return postgres_pod(
        client=unprivileged_client,
        namespace=unprivileged_model_namespace.name,
        label_selector=PRAXIS_POSTGRES_POD_LABEL_SELECTOR,
    )
