import uuid
from collections.abc import Generator
from typing import Any

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.evalhub import EvalHub
from ocp_resources.namespace import Namespace
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.route import Route
from ocp_resources.service import Service
from ocp_resources.service_account import ServiceAccount
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.model_explainability.evalhub.constants import (
    EVALHUB_HEALTH_PATH,
    EVALHUB_TENANT_LABEL_KEY,
    EVALHUB_VLLM_EMULATOR_PORT,
)
from tests.model_explainability.evalhub.utils import tenant_rbac_ready
from utilities.certificates_utils import create_ca_bundle_file
from utilities.constants import Labels, Protocols, Timeout
from utilities.infra import create_inference_token, create_ns

LOGGER = structlog.get_logger(name=__name__)


# ---------------------------------------------------------------------------
# EvalHub instance (shared across the class)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def evalhub_mt_cr(
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[EvalHub, Any, Any]:
    """Create an EvalHub CR for multi-tenancy tests.

    Uses a distinct name ('evalhub-mt') to avoid RoleBinding name collisions
    with the production EvalHub instance. The operator names tenant RoleBindings
    as '{instance.Name}-{ns}-job-config-rb' and uses Get-or-Create (not Update),
    so two instances named 'evalhub' would collide and the first one wins.
    """
    with EvalHub(
        client=admin_client,
        name="evalhub-mt",
        namespace=model_namespace.name,
        database={"type": "sqlite"},
        collections=["leaderboard-v2"],
        wait_for_resource=True,
    ) as evalhub:
        yield evalhub


@pytest.fixture(scope="class")
def evalhub_mt_deployment(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    evalhub_mt_cr: EvalHub,
) -> Deployment:
    """Wait for the EvalHub deployment to become available."""
    deployment = Deployment(
        client=admin_client,
        name=evalhub_mt_cr.name,
        namespace=model_namespace.name,
    )
    deployment.wait_for_replicas(timeout=Timeout.TIMEOUT_5MIN)
    return deployment


@pytest.fixture(scope="class")
def evalhub_mt_route(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    evalhub_mt_deployment: Deployment,
) -> Route:
    """Get the Route for the EvalHub service."""
    return Route(
        client=admin_client,
        name=evalhub_mt_deployment.name,
        namespace=model_namespace.name,
        ensure_exists=True,
    )


@pytest.fixture(scope="class")
def evalhub_mt_ca_bundle_file(
    admin_client: DynamicClient,
) -> str:
    """CA bundle file for verifying TLS on the EvalHub route."""
    return create_ca_bundle_file(client=admin_client)


@pytest.fixture(scope="class")
def evalhub_mt_ready(
    evalhub_mt_route: Route,
    evalhub_mt_ca_bundle_file: str,
) -> None:
    """Wait for the EvalHub service to respond via its route.

    The deployment may report ready replicas before the OpenShift router
    has fully configured the backend, causing 503 errors. This fixture
    polls the health endpoint until it responds successfully.
    """
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
# Tenant namespaces
# ---------------------------------------------------------------------------


@pytest.fixture(scope="class")
def tenant_a_namespace(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
) -> Generator[Namespace, Any, Any]:
    """Tenant namespace where the test user HAS access."""
    cls_name = request.cls.__name__.lower() if request.cls else "default"
    suffix = uuid.uuid4().hex[:6]
    name = f"test-evalhub-tenant-a-{cls_name}-{suffix}"
    with create_ns(
        admin_client=admin_client,
        name=name,
        labels={EVALHUB_TENANT_LABEL_KEY: "true"},
    ) as ns:
        yield ns


@pytest.fixture(scope="class")
def tenant_b_namespace(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
) -> Generator[Namespace, Any, Any]:
    """Tenant namespace where the test user does NOT have access."""
    cls_name = request.cls.__name__.lower() if request.cls else "default"
    suffix = uuid.uuid4().hex[:6]
    name = f"test-evalhub-tenant-b-{cls_name}-{suffix}"
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
def tenant_a_rbac_ready(
    admin_client: DynamicClient,
    tenant_a_namespace: Namespace,
    evalhub_mt_deployment: Deployment,
) -> None:
    """Wait for the operator to provision job RBAC in tenant-a.

    The operator watches for namespaces with the tenant label and
    creates jobs-writer + job-config RoleBindings. This fixture
    blocks until those RoleBindings exist.
    """
    try:
        for ready in TimeoutSampler(
            wait_timeout=120,
            sleep=5,
            func=tenant_rbac_ready,
            admin_client=admin_client,
            namespace=tenant_a_namespace.name,
        ):
            if ready:
                LOGGER.info(f"Operator RBAC provisioned in {tenant_a_namespace.name}")
                return
    except TimeoutExpiredError as err:
        msg = (
            f"Operator RBAC provision failed: RoleBindings, ServiceAccount, or service-CA ConfigMap"
            f" not found in namespace '{tenant_a_namespace.name}' within timeout"
        )
        LOGGER.error(msg)
        raise RuntimeError(msg) from err


# ---------------------------------------------------------------------------
# ServiceAccount and RBAC (only in tenant-a)
# ---------------------------------------------------------------------------

# Mirrors the user RBAC template from resources/evalhub-user-rbac-template.yaml.
# evaluations/collections/providers are virtual SAR resources — not real CRDs.
EVALHUB_USER_ROLE_RULES: list[dict[str, list[str]]] = [
    {
        "apiGroups": ["trustyai.opendatahub.io"],
        "resources": ["evaluations", "collections", "providers", "status-events"],
        "verbs": ["get", "list", "create", "update", "patch", "delete"],
    },
    {
        "apiGroups": ["mlflow.kubeflow.org"],
        "resources": ["experiments"],
        "verbs": ["create", "get"],
    },
]


@pytest.fixture(scope="class")
def tenant_a_service_account(
    admin_client: DynamicClient,
    tenant_a_namespace: Namespace,
) -> Generator[ServiceAccount, Any, Any]:
    """ServiceAccount in tenant-a for multi-tenancy tests."""
    with ServiceAccount(
        client=admin_client,
        name="evalhub-test-user",
        namespace=tenant_a_namespace.name,
        wait_for_resource=True,
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def tenant_a_evalhub_role(
    admin_client: DynamicClient,
    tenant_a_namespace: Namespace,
) -> Generator[Role, Any, Any]:
    """Role granting full EvalHub API access in tenant-a (virtual SAR resources)."""
    with Role(
        client=admin_client,
        name="evalhub-test-user-access",
        namespace=tenant_a_namespace.name,
        rules=EVALHUB_USER_ROLE_RULES,
        wait_for_resource=True,
    ) as role:
        yield role


@pytest.fixture(scope="class")
def tenant_a_evalhub_role_binding(
    admin_client: DynamicClient,
    tenant_a_namespace: Namespace,
    tenant_a_service_account: ServiceAccount,
    tenant_a_evalhub_role: Role,
) -> Generator[RoleBinding, Any, Any]:
    """RoleBinding granting the test SA EvalHub access in tenant-a only."""
    with RoleBinding(
        client=admin_client,
        name="evalhub-test-user-binding",
        namespace=tenant_a_namespace.name,
        subjects_kind="ServiceAccount",
        subjects_name=tenant_a_service_account.name,
        role_ref_kind="Role",
        role_ref_name=tenant_a_evalhub_role.name,
        wait_for_resource=True,
    ) as rb:
        yield rb


@pytest.fixture(scope="class")
def tenant_a_token(
    tenant_a_service_account: ServiceAccount,
    tenant_a_evalhub_role_binding: RoleBinding,
) -> str:
    """Bearer token for the test SA (has access to tenant-a, not tenant-b)."""
    return create_inference_token(model_service_account=tenant_a_service_account)


# ---------------------------------------------------------------------------
# vLLM emulator (deployed in tenant-a for job submission tests)
# ---------------------------------------------------------------------------

VLLM_EMULATOR: str = "vllm-emulator"
VLLM_EMULATOR_IMAGE: str = (
    "quay.io/trustyai_testing/vllm_emulator@sha256:c4bdd5bb93171dee5b4c8454f36d7c42b58b2a4ceb74f29dba5760ac53b5c12d"
)


@pytest.fixture(scope="class")
def evalhub_vllm_emulator_deployment(
    admin_client: DynamicClient,
    tenant_a_namespace: Namespace,
    tenant_a_rbac_ready: None,
) -> Generator[Deployment, Any, Any]:
    """Deploy the vLLM emulator in tenant-a.

    Depends on tenant_a_rbac_ready to ensure the operator has provisioned
    the jobs-writer and job-config RoleBindings before any job is submitted.
    """
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
def evalhub_vllm_emulator_service(
    admin_client: DynamicClient,
    tenant_a_namespace: Namespace,
    evalhub_vllm_emulator_deployment: Deployment,
) -> Generator[Service, Any, Any]:
    """Service fronting the vLLM emulator in tenant-a."""
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
