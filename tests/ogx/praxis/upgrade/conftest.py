from collections.abc import Generator

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.service import Service

from tests.ogx.praxis.upgrade.constants import (
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
    migration_target_secret_ref,
    postgres_pod,
    praxis_connection_string,
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
    pytestconfig: pytest.Config,
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
        ensure_exists=pytestconfig.option.post_upgrade,
    )
    if pytestconfig.option.post_upgrade:
        deployment.wait_for_replicas(deployed=True, timeout=240)
        yield deployment
        deployment.clean_up()
    else:
        with deployment:
            deployment.wait_for_replicas(deployed=True, timeout=240)
            yield deployment


@pytest.fixture(scope="class")
def praxis_postgres_service(
    pytestconfig: pytest.Config,
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
        ensure_exists=pytestconfig.option.post_upgrade,
        teardown=teardown_resources,
    )
    if pytestconfig.option.post_upgrade:
        yield service
        service.clean_up()
    else:
        with service:
            yield service


@pytest.fixture(scope="class")
def praxis_connection_string_secret(
    pytestconfig: pytest.Config,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    praxis_postgres_service: Service,
    teardown_resources: bool,
) -> Generator[Secret]:
    """Secret holding the Praxis DSN for `migrationJob.targetConnectionString`.

    Created before the upgrade, because the upgrade step references it by the
    name and key pinned in the praxis constants.
    """
    secret = Secret(
        client=unprivileged_client,
        namespace=unprivileged_model_namespace.name,
        name=PRAXIS_CONNECTION_SECRET_NAME,
        type="Opaque",
        string_data={
            PRAXIS_CONNECTION_SECRET_KEY: praxis_connection_string(namespace=unprivileged_model_namespace.name)
        },
        ensure_exists=pytestconfig.option.post_upgrade,
        teardown=teardown_resources,
    )
    if pytestconfig.option.post_upgrade:
        yield secret
        secret.clean_up()
    else:
        with secret:
            yield secret


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


@pytest.fixture(scope="class")
def configured_migration_target(ogx_server: OgxServer) -> dict[str, str]:
    """Skip unless the upgrade step configured the migration Job, whose completion the test waits for."""
    target_ref = migration_target_secret_ref(ogx_server=ogx_server)
    if target_ref is None:
        pytest.skip(
            f"OGXServer {ogx_server.name} has no spec.praxisMode.migrationJob.targetConnectionString; "
            "the upgrade job is expected to configure Praxis migration mode"
        )
    return target_ref
