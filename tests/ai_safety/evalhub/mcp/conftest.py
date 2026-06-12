import socket
from collections.abc import Generator
from typing import Any

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.custom_resource_definition import CustomResourceDefinition
from ocp_resources.deployment import Deployment
from ocp_resources.evalhub import EvalHub
from ocp_resources.namespace import Namespace
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.service_account import ServiceAccount
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_safety.evalhub.constants import EVALHUB_VLLM_EMULATOR_PORT
from tests.ai_safety.evalhub.kueue.constants import (
    VLLM_EMULATOR,
    VLLM_EMULATOR_IMAGE,
)
from tests.ai_safety.evalhub.mcp.constants import (
    EVALHUB_MCP_CR_NAME,
    EVALHUB_MCP_HEALTH_PATH,
)
from tests.ai_safety.evalhub.mcp.utils import (
    EvalHubMcpClient,
    build_mcp_proxy_role_rules,
    tenant_mcp_rbac_ready,
)
from utilities.certificates_utils import create_ca_bundle_file
from utilities.constants import Labels, Protocols, Timeout
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
    *,
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


def _is_evalhub_crd_available(admin_client: DynamicClient) -> bool:
    """Check if EvalHub CRD is installed on the cluster."""
    crd_name = "evalhubs.trustyai.opendatahub.io"
    try:
        crd = CustomResourceDefinition(client=admin_client, name=crd_name)
        return crd.exists
    except AttributeError, KeyError:
        return False


def _mcp_deployment_name(cr_name: str) -> str:
    return f"{cr_name}-mcp"


def _mcp_auth_secret_name(cr_name: str) -> str:
    return f"{cr_name}-mcp-token"


def _evalhub_service_account_name(cr_name: str) -> str:
    return f"{cr_name}-service"


@pytest.fixture(scope="class")
def evalhub_mcp_mt_cr(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    tenant_a_namespace: Namespace,
) -> Generator[EvalHub, Any, Any]:
    """Create an EvalHub CR with MCP enabled for integration tests."""
    if not _is_evalhub_crd_available(admin_client):
        pytest.fail(
            "EvalHub CRD 'evalhubs.trustyai.opendatahub.io' not available on this cluster. "
            "Install the TrustyAI/EvalHub operator first."
        )

    with EvalHub(
        client=admin_client,
        name=EVALHUB_MCP_CR_NAME,
        namespace=model_namespace.name,
        database={"type": "sqlite"},
        collections=["leaderboard-v2"],
        wait_for_resource=False,
    ) as evalhub:
        evalhub.to_dict()
        evalhub.res["spec"]["mcp"] = {
            "enabled": True,
            "replicas": 1,
            "env": [
                {
                    "name": "EVALHUB_TENANT",
                    "value": tenant_a_namespace.name,
                }
            ],
        }
        evalhub.create()
        evalhub.wait_for_resource()

        service_account = ServiceAccount(
            client=admin_client,
            name=_evalhub_service_account_name(EVALHUB_MCP_CR_NAME),
            namespace=model_namespace.name,
        )
        try:
            for _ in TimeoutSampler(
                wait_timeout=120,
                sleep=5,
                func=lambda: service_account.exists,
            ):
                if service_account.exists:
                    break
        except TimeoutExpiredError as err:
            raise RuntimeError(
                f"EvalHub service account '{service_account.name}' not created in {model_namespace.name}"
            ) from err

        token = create_inference_token(model_service_account=service_account)
        secret_name = _mcp_auth_secret_name(cr_name=EVALHUB_MCP_CR_NAME)
        with Secret(
            client=admin_client,
            name=secret_name,
            namespace=model_namespace.name,
            string_data={"token": token},
            wait_for_resource=True,
        ):
            evalhub.update(
                resource_dict={
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
                    }
                }
            )
            evalhub.wait_for_resource()
            yield evalhub


@pytest.fixture(scope="class")
def evalhub_mcp_mt_deployment(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    evalhub_mcp_mt_cr: EvalHub,
) -> Deployment:
    """Wait for the EvalHub MCP deployment to become available."""
    deployment = Deployment(
        client=admin_client,
        name=_mcp_deployment_name(EVALHUB_MCP_CR_NAME),
        namespace=model_namespace.name,
    )
    deployment.wait_for_replicas(timeout=Timeout.TIMEOUT_5MIN)
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
def evalhub_mcp_client(
    tenant_a_token: str,
    tenant_a_namespace: Namespace,
    evalhub_mcp_mt_route: Route,
    evalhub_mcp_mt_ca_bundle_file: str,
    evalhub_mcp_proxy_role_binding: RoleBinding,
    evalhub_mcp_mt_ready: None,
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


@pytest.fixture(scope="class")
def tenant_a_mcp_rbac_ready(
    admin_client: DynamicClient,
    tenant_a_namespace: Namespace,
    evalhub_mcp_mt_deployment: Deployment,
) -> None:
    """Wait for operator RBAC provisioned for the MCP EvalHub instance in tenant-a."""
    try:
        for ready in TimeoutSampler(
            wait_timeout=120,
            sleep=5,
            func=tenant_mcp_rbac_ready,
            admin_client=admin_client,
            namespace=tenant_a_namespace.name,
            evalhub_instance_name=EVALHUB_MCP_CR_NAME,
        ):
            if ready:
                LOGGER.info(f"Operator MCP RBAC provisioned in {tenant_a_namespace.name}")
                return
    except TimeoutExpiredError as err:
        msg = (
            f"Operator MCP RBAC not provisioned in '{tenant_a_namespace.name}' within timeout "
            f"for EvalHub instance '{EVALHUB_MCP_CR_NAME}'"
        )
        LOGGER.error(msg)
        raise RuntimeError(msg) from err


@pytest.fixture(scope="class")
def evalhub_mcp_vllm_emulator_deployment(
    admin_client: DynamicClient,
    tenant_a_namespace: Namespace,
    tenant_a_mcp_rbac_ready: None,
) -> Generator[Deployment, Any, Any]:
    """Deploy the vLLM emulator in tenant-a for MCP job submission tests."""
    label = {Labels.Openshift.APP: VLLM_EMULATOR}
    with Deployment(
        client=admin_client,
        namespace=tenant_a_namespace.name,
        name=VLLM_EMULATOR,
        label=label,
        selector={"matchLabels": label},
        template={
            "metadata": {
                "labels": label,
                "name": VLLM_EMULATOR,
            },
            "spec": {
                "containers": [
                    {
                        "image": VLLM_EMULATOR_IMAGE,
                        "name": VLLM_EMULATOR,
                        "ports": [{"containerPort": EVALHUB_VLLM_EMULATOR_PORT, "protocol": Protocols.TCP}],
                        "readinessProbe": {
                            "tcpSocket": {"port": EVALHUB_VLLM_EMULATOR_PORT},
                            "initialDelaySeconds": 5,
                            "periodSeconds": 5,
                            "timeoutSeconds": 3,
                            "failureThreshold": 6,
                        },
                        "securityContext": {
                            "allowPrivilegeEscalation": False,
                            "capabilities": {"drop": ["ALL"]},
                            "seccompProfile": {"type": "RuntimeDefault"},
                        },
                    }
                ]
            },
        },
        replicas=1,
    ) as deployment:
        deployment.wait_for_replicas(timeout=Timeout.TIMEOUT_5MIN)
        yield deployment


@pytest.fixture(scope="class")
def evalhub_mcp_vllm_emulator_service(
    admin_client: DynamicClient,
    tenant_a_namespace: Namespace,
    evalhub_mcp_vllm_emulator_deployment: Deployment,
) -> Generator[Service, Any, Any]:
    """Service fronting the vLLM emulator in tenant-a for MCP job tests."""
    with Service(
        client=admin_client,
        namespace=tenant_a_namespace.name,
        name=f"{VLLM_EMULATOR}-service",
        ports=[
            {
                "name": f"{VLLM_EMULATOR}-endpoint",
                "port": EVALHUB_VLLM_EMULATOR_PORT,
                "protocol": Protocols.TCP,
                "targetPort": EVALHUB_VLLM_EMULATOR_PORT,
            }
        ],
        selector={Labels.Openshift.APP: VLLM_EMULATOR},
    ) as service:
        yield service
