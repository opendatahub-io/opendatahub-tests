import uuid
from collections.abc import Generator
from typing import Any

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.route import Route
from ocp_resources.service import Service
from ocp_resources.service_account import ServiceAccount
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_safety.evalhub.constants import (
    EVALHUB_HEALTH_PATH,
    EVALHUB_TENANT_LABEL_KEY,
    EVALHUB_VLLM_EMULATOR_PORT,
)
from tests.ai_safety.evalhub.utils import tenant_rbac_ready
from utilities.constants import Labels, Protocols, Timeout
from utilities.infra import create_inference_token, create_ns

LOGGER = structlog.get_logger(name=__name__)


# ---------------------------------------------------------------------------
# Note: evalhub_mt_cr, evalhub_mt_deployment, evalhub_mt_route, and
# evalhub_mt_ca_bundle_file fixtures are defined in ../conftest.py (parent)
# and shared across all evalhub test subdirectories.
# ---------------------------------------------------------------------------


EVALHUB_MCP_USER_ROLE_RULES: list[dict[str, list[str]]] = [
    {
        "apiGroups": ["trustyai.opendatahub.io"],
        "resources": ["evaluations", "collections", "providers"],
        "verbs": ["get", "list", "create", "update", "delete"],
    },
]


# ---------------------------------------------------------------------------
# EvalHub readiness
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def evalhub_mcp_ready(
    evalhub_mt_route: Route,
    evalhub_mt_ca_bundle_file: str,
) -> None:
    """Wait for the EvalHub service to respond via its route."""
    url = f"https://{evalhub_mt_route.host}{EVALHUB_HEALTH_PATH}"
    try:
        for sample in TimeoutSampler(
            wait_timeout=120,
            sleep=5,
            func=lambda: requests.get(url, verify=evalhub_mt_ca_bundle_file, timeout=10),
            exceptions_dict={Exception: []},
        ):
            if sample.ok:
                LOGGER.info(f"EvalHub at {evalhub_mt_route.host} is healthy")
                return
    except TimeoutExpiredError as err:
        raise RuntimeError(f"EvalHub at {evalhub_mt_route.host} did not become healthy within 120s") from err


# ---------------------------------------------------------------------------
# Tenant namespace
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def mcp_tenant_namespace(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
) -> Generator[Namespace, Any, Any]:
    """Tenant namespace for MCP evaluation tests."""
    cls_name = request.cls.__name__.lower() if request.cls else "default"
    suffix = uuid.uuid4().hex[:6]
    name = f"test-evalhub-mcp-{cls_name}-{suffix}"
    with create_ns(
        admin_client=admin_client,
        name=name,
        labels={EVALHUB_TENANT_LABEL_KEY: "true"},
    ) as ns:
        yield ns


# ---------------------------------------------------------------------------
# Wait for operator to provision tenant RBAC
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def mcp_tenant_rbac_ready(
    admin_client: DynamicClient,
    mcp_tenant_namespace: Namespace,
    evalhub_mt_deployment: Deployment,
) -> None:
    """Wait for the operator to provision job RBAC in the MCP tenant namespace."""
    try:
        for ready in TimeoutSampler(
            wait_timeout=120,
            sleep=5,
            func=tenant_rbac_ready,
            admin_client=admin_client,
            namespace=mcp_tenant_namespace.name,
        ):
            if ready:
                LOGGER.info(f"Operator RBAC provisioned in {mcp_tenant_namespace.name}")
                return
    except TimeoutExpiredError as err:
        msg = (
            f"Operator RBAC provision failed: RoleBindings, ServiceAccount, or service-CA ConfigMap"
            f" not found in namespace '{mcp_tenant_namespace.name}' within timeout"
        )
        LOGGER.error(msg)
        raise RuntimeError(msg) from err


# ---------------------------------------------------------------------------
# ServiceAccount and RBAC
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def mcp_tenant_service_account(
    admin_client: DynamicClient,
    mcp_tenant_namespace: Namespace,
) -> Generator[ServiceAccount, Any, Any]:
    """ServiceAccount for MCP evaluation tests."""
    with ServiceAccount(
        client=admin_client,
        name="evalhub-mcp-test-user",
        namespace=mcp_tenant_namespace.name,
        wait_for_resource=True,
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def mcp_tenant_role(
    admin_client: DynamicClient,
    mcp_tenant_namespace: Namespace,
) -> Generator[Role, Any, Any]:
    """Role granting EvalHub evaluation permissions for MCP tests."""
    with Role(
        client=admin_client,
        name="evalhub-mcp-test-access",
        namespace=mcp_tenant_namespace.name,
        rules=EVALHUB_MCP_USER_ROLE_RULES,
        wait_for_resource=True,
    ) as role:
        yield role


@pytest.fixture(scope="class")
def mcp_tenant_role_binding(
    admin_client: DynamicClient,
    mcp_tenant_namespace: Namespace,
    mcp_tenant_service_account: ServiceAccount,
    mcp_tenant_role: Role,
) -> Generator[RoleBinding, Any, Any]:
    """RoleBinding granting the MCP test SA EvalHub access."""
    with RoleBinding(
        client=admin_client,
        name="evalhub-mcp-test-binding",
        namespace=mcp_tenant_namespace.name,
        subjects_kind="ServiceAccount",
        subjects_name=mcp_tenant_service_account.name,
        role_ref_kind="Role",
        role_ref_name=mcp_tenant_role.name,
        wait_for_resource=True,
    ) as rb:
        yield rb


@pytest.fixture(scope="class")
def mcp_tenant_token(
    mcp_tenant_service_account: ServiceAccount,
    mcp_tenant_role_binding: RoleBinding,
) -> str:
    """Bearer token for the MCP test ServiceAccount."""
    return create_inference_token(model_service_account=mcp_tenant_service_account)


# ---------------------------------------------------------------------------
# vLLM emulator
# ---------------------------------------------------------------------------

VLLM_EMULATOR: str = "vllm-emulator"
VLLM_EMULATOR_IMAGE: str = (
    "quay.io/trustyai_testing/vllm_emulator@sha256:c4bdd5bb93171dee5b4c8454f36d7c42b58b2a4ceb74f29dba5760ac53b5c12d"
)


@pytest.fixture(scope="class")
def mcp_vllm_emulator_deployment(
    admin_client: DynamicClient,
    mcp_tenant_namespace: Namespace,
    mcp_tenant_rbac_ready: None,
) -> Generator[Deployment, Any, Any]:
    """Deploy the vLLM emulator in the MCP tenant namespace."""
    label = {Labels.Openshift.APP: VLLM_EMULATOR}
    with Deployment(
        client=admin_client,
        namespace=mcp_tenant_namespace.name,
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
def mcp_vllm_emulator_service(
    admin_client: DynamicClient,
    mcp_tenant_namespace: Namespace,
    mcp_vllm_emulator_deployment: Deployment,
) -> Generator[Service, Any, Any]:
    """Service fronting the vLLM emulator for MCP tests."""
    with Service(
        client=admin_client,
        namespace=mcp_tenant_namespace.name,
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
