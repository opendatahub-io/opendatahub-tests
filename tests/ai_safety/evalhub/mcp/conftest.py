import socket
from collections.abc import Generator
from typing import Any

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.evalhub import EvalHub
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_safety.evalhub.constants import EVALHUB_USER_ROLE_RULES
from tests.ai_safety.evalhub.mcp.constants import (
    EVALHUB_MCP_APP_LABEL,
    EVALHUB_MCP_COMPONENT_LABEL,
    EVALHUB_MCP_CR_NAME,
    EVALHUB_MCP_HEALTH_PATH,
)
from tests.ai_safety.evalhub.mcp.utils import (
    EvalHubMcpClient,
    build_mcp_proxy_role_rules,
)
from tests.ai_safety.evalhub.utils import is_evalhub_crd_available, wait_for_service_account
from utilities.certificates_utils import create_ca_bundle_file
from utilities.infra import create_inference_token

LOGGER = structlog.get_logger(name=__name__)


class _TransientEvalhubMcpHealthError(Exception):
    """Recoverable failure while polling the EvalHub MCP health endpoint."""


_TRANSIENT_MCP_HEALTH_REQUEST_EXCEPTIONS = (
    requests.exceptions.ConnectTimeout,
    requests.exceptions.ReadTimeout,
)
_TRANSIENT_MCP_HEALTH_EXCEPTIONS = {_TransientEvalhubMcpHealthError: []}


def _is_dns_resolution_error(err: BaseException) -> bool:
    """Return True when the exception chain includes a DNS resolution failure."""
    exc: BaseException | None = err
    while exc is not None:
        if isinstance(exc, socket.gaierror):
            return True
        exc = exc.__cause__
    return False


def _probe_evalhub_mcp_health(
    url: str,
    host: str,
    ca_bundle_file: str,
) -> requests.Response:
    """GET the MCP health endpoint, retrying only on transient network failures."""
    try:
        return requests.get(url, verify=ca_bundle_file, timeout=10)
    except requests.exceptions.ConnectionError as err:
        if isinstance(err, requests.exceptions.SSLError) or _is_dns_resolution_error(err):
            raise
        LOGGER.warning(f"Transient error checking EvalHub MCP health at {host}: {err}")
        raise _TransientEvalhubMcpHealthError(str(err)) from err
    except _TRANSIENT_MCP_HEALTH_REQUEST_EXCEPTIONS as err:
        LOGGER.warning(f"Transient error checking EvalHub MCP health at {host}: {err}")
        raise _TransientEvalhubMcpHealthError(str(err)) from err


def _mcp_deployment_name(cr_name: str) -> str:
    return f"{cr_name}-mcp"


def _mcp_auth_secret_name(cr_name: str) -> str:
    return f"{cr_name}-mcp-token"


def _evalhub_service_account_name(cr_name: str) -> str:
    return f"{cr_name}-service"


def _wait_for_deployment_rollout(deployment: Deployment, timeout: int = 300) -> None:
    """Wait until all replicas are running the latest pod template.

    ``wait_for_replicas`` passes as soon as *any* replicas are ready, which
    can be satisfied by old pods during a rolling update. This helper polls
    until ``updatedReplicas == spec.replicas`` and no unavailable replicas
    remain, guaranteeing all pods reflect the latest Deployment spec.

    During a rolling update, ``maxSurge`` can create a temporary extra pod
    (e.g. 1 replica -> surge to 2) before the old pod terminates. Checking
    ``updatedReplicas``/``unavailableReplicas`` alone can pass while that old
    surge pod is still being torn down, so ``replicas`` (the total pod count)
    must also have settled back down to ``updatedReplicas``.
    """
    for sample in TimeoutSampler(
        wait_timeout=timeout,
        sleep=5,
        func=lambda: deployment.instance.status,
    ):
        desired = deployment.instance.spec.replicas or 1
        updated = getattr(sample, "updatedReplicas", None) or 0
        total = getattr(sample, "replicas", None) or 0
        unavailable = getattr(sample, "unavailableReplicas", None) or 0
        if updated >= desired and total <= updated and unavailable == 0:
            return


def _wait_for_stable_pod_count(
    admin_client: DynamicClient,
    namespace: str,
    instance_name: str,
    desired: int,
    timeout: int = 120,
    stable_checks: int = 3,
    interval: int = 5,
) -> None:
    """Wait until the MCP pod count matches ``desired`` for several consecutive polls.

    The EvalHub operator can re-reconcile the deployment shortly after a rollout
    finishes (e.g. following an authSecret patch), starting a second rolling
    update whose surge pod appears after ``_wait_for_deployment_rollout`` already
    observed a settled status. Polling the actual pod list over a short window
    catches that trailing rollout instead of trusting a single status snapshot.
    """
    label_selector = f"app={EVALHUB_MCP_APP_LABEL},instance={instance_name},component={EVALHUB_MCP_COMPONENT_LABEL}"
    consecutive_matches = 0
    for _ in TimeoutSampler(wait_timeout=timeout, sleep=interval, func=lambda: True):
        pods = list(Pod.get(client=admin_client, namespace=namespace, label_selector=label_selector))
        if len(pods) == desired and all(pod.instance.status.phase == Pod.Status.RUNNING for pod in pods):
            consecutive_matches += 1
            if consecutive_matches >= stable_checks:
                return
        else:
            consecutive_matches = 0


@pytest.fixture(scope="class")
def evalhub_tenant_rbac_instance_name() -> str:  # noqa: UFN001
    """EvalHub CR name used when waiting for operator job RBAC in tenant namespaces."""
    return EVALHUB_MCP_CR_NAME


@pytest.fixture(scope="class")
def evalhub_tenant_deployment(evalhub_mcp_mt_deployment: Deployment) -> Deployment:  # noqa: UFN001
    """EvalHub deployment whose operator RBAC must be ready in tenant namespaces."""
    return evalhub_mcp_mt_deployment


@pytest.fixture(scope="class")
def evalhub_mcp_mt_cr(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    tenant_a_namespace: Namespace,
) -> Generator[EvalHub, Any, Any]:
    """Create an EvalHub CR with MCP enabled for integration tests."""
    if not is_evalhub_crd_available(admin_client):
        pytest.fail(
            "EvalHub CRD 'evalhubs.trustyai.opendatahub.io' not available on this cluster. "
            "Install the TrustyAI/EvalHub operator first."
        )

    # kind_dict is required: EvalHub's generated to_dict() has no "mcp" kwarg and
    # resets res["spec"] from its known attributes on every create() call.
    with EvalHub(
        client=admin_client,
        kind_dict={
            "apiVersion": f"{EvalHub.api_group}/v1",
            "kind": "EvalHub",
            "metadata": {
                "name": EVALHUB_MCP_CR_NAME,
                "namespace": model_namespace.name,
            },
            "spec": {
                "database": {"type": "sqlite"},
                "collections": ["leaderboard-v2"],
                "mcp": {
                    "enabled": True,
                    "replicas": 1,
                    "env": [
                        {
                            "name": "EVALHUB_TENANT",
                            "value": tenant_a_namespace.name,
                        }
                    ],
                },
            },
        },
        wait_for_resource=False,
    ) as evalhub:
        # Poll until the EvalHub operator reports the CR as ready.
        # Pending and None are expected and should not stop polling.
        for sample in TimeoutSampler(wait_timeout=300, sleep=2, func=lambda: evalhub.instance.status):
            if sample is None:
                continue
            if sample.get("ready") == "True":
                break
            phase = sample.get("phase", "")
            if phase == "Error":
                mcp_status = sample.get("mcp", {})
                pytest.fail(
                    f"EvalHub entered Error phase during setup.\n"
                    f"  Top-level status: {sample}\n"
                    f"  MCP sub-status:   {mcp_status}"
                )
        yield evalhub


@pytest.fixture(scope="class")
def evalhub_mcp_service_account(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    evalhub_mcp_mt_cr: EvalHub,
) -> ServiceAccount:
    """Wait for the operator-created EvalHub service account in the model namespace."""
    return wait_for_service_account(
        admin_client=admin_client,
        namespace=model_namespace.name,
        sa_name=_evalhub_service_account_name(EVALHUB_MCP_CR_NAME),
        timeout=120,
    )


@pytest.fixture(scope="class")
def evalhub_mcp_mt_cr_with_auth(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    tenant_a_namespace: Namespace,
    evalhub_mcp_mt_cr: EvalHub,
    evalhub_mcp_service_account: ServiceAccount,
) -> Generator[EvalHub, Any, Any]:
    """Patch the EvalHub CR with MCP auth secret configuration."""
    token = create_inference_token(model_service_account=evalhub_mcp_service_account)
    secret_name = _mcp_auth_secret_name(cr_name=EVALHUB_MCP_CR_NAME)
    with Secret(
        client=admin_client,
        name=secret_name,
        namespace=model_namespace.name,
        string_data={"token": token},
        wait_for_resource=False,
    ):
        evalhub_mcp_mt_cr.update(
            resource_dict={
                "metadata": {
                    "name": EVALHUB_MCP_CR_NAME,
                    "namespace": model_namespace.name,
                },
                "spec": {
                    "mcp": {
                        "enabled": True,
                        "replicas": 1,
                        "authSecret": secret_name,
                        "env": [
                            {
                                "name": "EVALHUB_TENANT",
                                "value": tenant_a_namespace.name,
                            }
                        ],
                    }
                },
            }
        )
        # Poll until the operator finishes reconciling the authSecret patch.
        # .wait() only checks object existence — not operator readiness.
        for sample in TimeoutSampler(wait_timeout=300, sleep=2, func=lambda: evalhub_mcp_mt_cr.instance.status):
            if sample is None:
                continue
            if sample.get("ready") == "True":
                break
            phase = sample.get("phase", "")
            if phase == "Error":
                mcp_status = sample.get("mcp", {})
                pytest.fail(
                    f"EvalHub entered Error phase after authSecret patch.\n"
                    f"  Top-level status: {sample}\n"
                    f"  MCP sub-status:   {mcp_status}"
                )
        yield evalhub_mcp_mt_cr


@pytest.fixture(scope="class")
def evalhub_mcp_mt_deployment(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    evalhub_mcp_mt_cr_with_auth: EvalHub,
) -> Deployment:
    """Wait for the EvalHub MCP deployment rollout to complete."""
    deployment = Deployment(
        client=admin_client,
        name=_mcp_deployment_name(EVALHUB_MCP_CR_NAME),
        namespace=model_namespace.name,
    )
    deployment.wait_for_replicas(timeout=300)
    _wait_for_deployment_rollout(deployment=deployment, timeout=300)
    _wait_for_stable_pod_count(
        admin_client=admin_client,
        namespace=model_namespace.name,
        instance_name=EVALHUB_MCP_CR_NAME,
        desired=deployment.instance.spec.replicas or 1,
        timeout=120,
    )
    return deployment


@pytest.fixture(scope="class")
def evalhub_mcp_mt_route(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    evalhub_mcp_mt_deployment: Deployment,
) -> Route:
    """Get the Route for the EvalHub MCP service."""
    return Route(
        client=admin_client,
        name=_mcp_deployment_name(EVALHUB_MCP_CR_NAME),
        namespace=model_namespace.name,
        ensure_exists=True,
    )


@pytest.fixture(scope="class")
def evalhub_mcp_mt_ca_bundle_file(
    admin_client: DynamicClient,
) -> str:
    """CA bundle file for verifying TLS on the EvalHub MCP route."""
    return create_ca_bundle_file(client=admin_client)


@pytest.fixture(scope="class")
def evalhub_mcp_mt_ready(
    evalhub_mcp_mt_route: Route,
    evalhub_mcp_mt_ca_bundle_file: str,
) -> None:
    """Wait until the MCP health endpoint responds on the route."""
    url = f"https://{evalhub_mcp_mt_route.host}{EVALHUB_MCP_HEALTH_PATH}"
    host = evalhub_mcp_mt_route.host
    try:
        for sample in TimeoutSampler(
            wait_timeout=120,
            sleep=5,
            func=lambda: _probe_evalhub_mcp_health(
                url=url,
                host=host,
                ca_bundle_file=evalhub_mcp_mt_ca_bundle_file,
            ),
            exceptions_dict=_TRANSIENT_MCP_HEALTH_EXCEPTIONS,
        ):
            if sample.ok:
                LOGGER.info(f"EvalHub MCP at {host} is healthy")
                return
    except TimeoutExpiredError as err:
        if err.last_exp is not None:
            raise err.last_exp from err
        raise RuntimeError(f"EvalHub MCP at {host} did not become healthy within 120s") from err


@pytest.fixture(scope="class")
def evalhub_mcp_proxy_role(
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[Role, Any, Any]:
    """Role in the EvalHub namespace granting evalhubs/proxy access to the MCP instance."""
    with Role(
        client=admin_client,
        name="evalhub-mcp-proxy-access",
        namespace=model_namespace.name,
        rules=build_mcp_proxy_role_rules(evalhub_instance_name=EVALHUB_MCP_CR_NAME),
        wait_for_resource=True,
    ) as role:
        yield role


@pytest.fixture(scope="class")
def evalhub_mcp_proxy_role_binding(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    tenant_a_service_account: ServiceAccount,
    evalhub_mcp_proxy_role: Role,
) -> Generator[RoleBinding, Any, Any]:
    """Bind MCP proxy access to the tenant-a test ServiceAccount."""
    with RoleBinding(
        client=admin_client,
        name="evalhub-mcp-proxy-binding",
        namespace=model_namespace.name,
        subjects_kind="ServiceAccount",
        subjects_name=tenant_a_service_account.name,
        subjects_namespace=tenant_a_service_account.namespace,
        role_ref_kind="Role",
        role_ref_name=evalhub_mcp_proxy_role.name,
        wait_for_resource=True,
    ) as binding:
        yield binding


@pytest.fixture(scope="class")
def mcp_server_tenant_rbac(
    admin_client: DynamicClient,
    tenant_a_namespace: Namespace,
    model_namespace: Namespace,
    evalhub_mcp_mt_deployment: Deployment,
) -> Generator[RoleBinding, Any, Any]:
    """Grant MCP server service account access to tenant namespace EvalHub API resources.

    This is a workaround until the EvalHub operator automatically provisions these permissions.
    The MCP server needs to access evaluations, providers, benchmarks, and collections in the
    tenant namespace on behalf of authenticated users.
    """
    mcp_server_sa_name = _evalhub_service_account_name(cr_name=EVALHUB_MCP_CR_NAME)

    # Create role in tenant namespace granting EvalHub API access
    with (
        Role(
            client=admin_client,
            name=f"{EVALHUB_MCP_CR_NAME}-server-access",
            namespace=tenant_a_namespace.name,
            rules=EVALHUB_USER_ROLE_RULES,  # Same permissions as test user
            wait_for_resource=True,
        ) as role,
        RoleBinding(
            client=admin_client,
            name=f"{EVALHUB_MCP_CR_NAME}-server-binding",
            namespace=tenant_a_namespace.name,
            subjects_kind="ServiceAccount",
            subjects_name=mcp_server_sa_name,
            subjects_namespace=model_namespace.name,
            role_ref_kind="Role",
            role_ref_name=role.name,
            wait_for_resource=True,
        ) as binding,
    ):
        yield binding


@pytest.fixture(scope="class")
def evalhub_mcp_client(
    tenant_a_token: str,
    tenant_a_namespace: Namespace,
    evalhub_mcp_mt_route: Route,
    evalhub_mcp_mt_ca_bundle_file: str,
    evalhub_mcp_proxy_role_binding: RoleBinding,
    evalhub_mcp_mt_ready: None,
    mcp_server_tenant_rbac: RoleBinding,
) -> EvalHubMcpClient:
    """Authenticated MCP client for tenant-a."""
    client = EvalHubMcpClient(
        host=evalhub_mcp_mt_route.host,
        token=tenant_a_token,
        ca_bundle_file=evalhub_mcp_mt_ca_bundle_file,
        tenant=tenant_a_namespace.name,
    )
    client.initialize()
    return client
