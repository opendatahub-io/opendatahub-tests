from collections.abc import Callable, Generator
from typing import Any

import httpx
import pytest
import requests
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ogx_client import OgxClient

from tests.ogx.constants import OGX_CLIENT_VERIFY_SSL
from tests.ogx.praxis.constants import (
    CONVERSATIONS_API_PATH,
    FILES_API_PATH,
    OGX_CONNECTION_SECRET_KEY,
    OGX_CONNECTION_SECRET_NAME,
    OGX_POSTGRES_POD_LABEL_SELECTOR,
    POSTGRES_PORT,
    PRAXIS_CONNECTION_SECRET_KEY,
    PRAXIS_CONNECTION_SECRET_NAME,
    PRAXIS_GATEWAY_CONFIG_KEY,
    PRAXIS_GATEWAY_CONFIG_SECRET_NAME,
    PRAXIS_GATEWAY_DEPLOYMENT_NAME,
    PRAXIS_GATEWAY_POD_LABELS,
    PRAXIS_GATEWAY_PORT,
    PRAXIS_GATEWAY_READY_TIMEOUT,
    PRAXIS_GATEWAY_SERVICE_NAME,
    PRAXIS_POSTGRES_APP_LABEL,
    PRAXIS_POSTGRES_DATABASE,
    PRAXIS_POSTGRES_DEPLOYMENT_NAME,
    PRAXIS_POSTGRES_POD_LABEL_SELECTOR,
    PRAXIS_POSTGRES_SERVICE_NAME,
    REQUEST_TIMEOUT_SECONDS,
    RESPONSES_API_PATH,
    VECTOR_STORES_API_PATH,
)
from tests.ogx.praxis.migration import run_two_step_migration
from tests.ogx.praxis.server_config import build_praxis_gateway_config
from tests.ogx.praxis.utils import (
    gateway_base_url,
    get_praxis_gateway_deployment_template,
    ogx_connection_string,
    postgres_pod,
    praxis_api_url,
    praxis_connection_string,
    praxis_gateway_base_url,
    praxis_http_route,
)
from tests.ogx.utils import get_postgres_deployment_template
from utilities.infra import get_openshift_token
from utilities.resources.http_route import HTTPRoute
from utilities.resources.ogx_server import OgxServer


@pytest.fixture
def tenant_authorization_header(unprivileged_client: DynamicClient) -> dict[str, str]:
    """Authorization header carrying the OpenShift token of the non-admin tenant."""
    return {
        "Authorization": f"Bearer {get_openshift_token(client=unprivileged_client)}",
        "Content-Type": "application/json",
    }


@pytest.fixture
def request_session() -> Generator[requests.Session, Any, Any]:
    """HTTP session for requests that must bypass the OGX SDK and its authentication."""
    session = requests.Session()
    session.verify = OGX_CLIENT_VERIFY_SSL
    yield session
    session.close()


@pytest.fixture(scope="class")
def praxis_client(admin_client: DynamicClient, unprivileged_client: DynamicClient) -> Generator[OgxClient, Any, Any]:
    """OgxClient bound to the external Gateway hostname that Praxis serves.

    The shared `ogx_client` addresses the OGX Service directly through an OpenShift
    Route, so it reaches OGX whatever the external routing does. Assertions about
    Praxis-era behaviour must traverse the Gateway -> Praxis -> OGX path instead.

    The two clients play different roles. The api_key comes from `unprivileged_client`
    because the namespace, the OGXServer and every resource seeded before the upgrade
    belong to the non-admin tenant, and Praxis scopes resources per tenant: reading
    them back with an admin token would be a cross-identity read. The route lookups
    use `admin_client` because resolving the Gateway hostname needs cluster-wide
    HTTPRoute read access, which the tenant does not have.
    """
    http_client = httpx.Client(verify=OGX_CLIENT_VERIFY_SSL, timeout=REQUEST_TIMEOUT_SECONDS)
    try:
        yield OgxClient(
            base_url=praxis_gateway_base_url(
                client=admin_client,
                paths=(RESPONSES_API_PATH, FILES_API_PATH, VECTOR_STORES_API_PATH, CONVERSATIONS_API_PATH),
            ),
            api_key=get_openshift_token(client=unprivileged_client),
            http_client=http_client,
            timeout=REQUEST_TIMEOUT_SECONDS,
            max_retries=0,
        )
    finally:
        http_client.close()


@pytest.fixture(scope="class")
def praxis_responses_http_route(admin_client: DynamicClient) -> HTTPRoute:
    """HTTPRoute publishing the Responses API at the public Gateway boundary."""
    return praxis_http_route(client=admin_client, path=RESPONSES_API_PATH)


@pytest.fixture(scope="class")
def praxis_files_http_route(admin_client: DynamicClient) -> HTTPRoute:
    """HTTPRoute publishing the Files API at the public Gateway boundary."""
    return praxis_http_route(client=admin_client, path=FILES_API_PATH)


@pytest.fixture(scope="class")
def praxis_responses_url(praxis_responses_http_route: HTTPRoute) -> str:
    """Public URL of the Responses API at the Praxis Gateway boundary."""
    return f"{gateway_base_url(http_route=praxis_responses_http_route)}{RESPONSES_API_PATH}"


@pytest.fixture(scope="class")
def praxis_files_url(praxis_files_http_route: HTTPRoute) -> str:
    """Public URL of the Files API at the Praxis Gateway boundary."""
    return f"{gateway_base_url(http_route=praxis_files_http_route)}{FILES_API_PATH}"


@pytest.fixture(scope="class")
def praxis_vector_stores_url(admin_client: DynamicClient) -> str:
    """Public URL of the Vector Stores API at the Praxis Gateway boundary."""
    return praxis_api_url(client=admin_client, path=VECTOR_STORES_API_PATH)


@pytest.fixture(scope="class")
def praxis_conversations_url(admin_client: DynamicClient) -> str:
    """Public URL of the Conversations API at the Praxis Gateway boundary."""
    return praxis_api_url(client=admin_client, path=CONVERSATIONS_API_PATH)


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
        label={"ogx.io/watch": "true"},
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
        label={"ogx.io/watch": "true"},
        type="Opaque",
        string_data={
            PRAXIS_CONNECTION_SECRET_KEY: praxis_connection_string(namespace=unprivileged_model_namespace.name)
        },
        teardown=teardown_resources,
    )
    with secret:
        yield secret


@pytest.fixture(scope="class")
def praxis_gateway_config_secret(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    praxis_postgres_service: Service,
    teardown_resources: bool,
) -> Generator[Secret]:
    """Secret holding the configuration of the Praxis gateway."""
    with Secret(
        client=unprivileged_client,
        namespace=unprivileged_model_namespace.name,
        name=PRAXIS_GATEWAY_CONFIG_SECRET_NAME,
        type="Opaque",
        string_data={
            PRAXIS_GATEWAY_CONFIG_KEY: build_praxis_gateway_config(namespace=unprivileged_model_namespace.name),
        },
        teardown=teardown_resources,
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def praxis_gateway_deployment(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    praxis_gateway_config_secret: Secret,
    teardown_resources: bool,
) -> Generator[Deployment]:
    """Deploy a Praxis gateway in front of the target database."""
    deployment = Deployment(
        client=unprivileged_client,
        namespace=unprivileged_model_namespace.name,
        name=PRAXIS_GATEWAY_DEPLOYMENT_NAME,
        replicas=1,
        selector={"matchLabels": PRAXIS_GATEWAY_POD_LABELS},
        template=get_praxis_gateway_deployment_template(),
        teardown=teardown_resources,
    )
    with deployment:
        deployment.wait_for_replicas(deployed=True, timeout=PRAXIS_GATEWAY_READY_TIMEOUT)
        yield deployment


@pytest.fixture(scope="class")
def praxis_gateway_service(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    praxis_gateway_deployment: Deployment,
    teardown_resources: bool,
) -> Generator[Service]:
    """Service fronting the Praxis gateway."""
    with Service(
        client=unprivileged_client,
        namespace=unprivileged_model_namespace.name,
        name=PRAXIS_GATEWAY_SERVICE_NAME,
        ports=[{"port": PRAXIS_GATEWAY_PORT, "targetPort": PRAXIS_GATEWAY_PORT}],
        selector=PRAXIS_GATEWAY_POD_LABELS,
        wait_for_resource=True,
        teardown=teardown_resources,
    ) as service:
        yield service


@pytest.fixture(scope="class")
def praxis_gateway_route(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    praxis_gateway_service: Service,
    teardown_resources: bool,
) -> Generator[Route]:
    """Route publishing the Praxis gateway, so the schema probes can reach it."""
    with Route(
        client=unprivileged_client,
        namespace=unprivileged_model_namespace.name,
        name=PRAXIS_GATEWAY_SERVICE_NAME,
        service=praxis_gateway_service.name,
        wait_for_resource=True,
        teardown=teardown_resources,
    ) as route:
        ResourceEditor(
            patches={
                route: {
                    "spec": {
                        "tls": {"termination": "edge", "insecureEdgeTerminationPolicy": "Redirect"},
                    }
                }
            }
        ).update()
        yield route


@pytest.fixture(scope="class")
def praxis_migration_runner(
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    ogx_server: OgxServer,
    ogx_connection_string_secret: Secret,
    praxis_connection_string_secret: Secret,
    praxis_gateway_route: Route,
    ogx_postgres_pod: Pod,
) -> Callable[[], OgxServer]:
    """Return a runner that enables Praxis mode and waits for database migration.

    Praxis mode and the migration are enabled in two steps. Enabling Praxis mode
    makes the operator rewrite the OGX config as multi-tenant, which OGX applies by
    adding `tenant_id` to its tables on its next start; the migration reads that
    column. `migrationJob` is left out of the first patch, which the API supports
    (the Job is created only once the field is set), so the migration starts
    against a source that already carries the column.
    """

    def _run_migration() -> OgxServer:
        # return run_migration(
        #     ogx_server=ogx_server,
        #     unprivileged_client=unprivileged_client,
        #     ogx_connection_string_name=ogx_connection_string_secret.name,
        #     praxis_connection_string_name=praxis_connection_string_secret.name,
        # )

        return run_two_step_migration(
            ogx_server=ogx_server,
            unprivileged_client=unprivileged_client,
            ogx_postgres_pod=ogx_postgres_pod,
            ogx_connection_string_name=ogx_connection_string_secret.name,
            praxis_connection_string_name=praxis_connection_string_secret.name,
            namespace=unprivileged_model_namespace.name,
        )

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
