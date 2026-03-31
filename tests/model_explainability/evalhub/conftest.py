from collections.abc import Generator
from contextlib import ExitStack
from typing import Any
import shlex

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.route import Route
from ocp_resources.service_account import ServiceAccount
from pyhelper_utils.shell import run_command

from tests.model_explainability.evalhub.constants import (
    EVALHUB_EVALUATOR_ROLE,
    EVALHUB_PROVIDERS_ACCESS_CLUSTER_ROLE,
    EVALHUB_TENANT_LABEL,
    TENANT_A_NAME,
    TENANT_A_SA_NAME,
    TENANT_B_NAME,
    TENANT_B_SA_NAME,
    TENANT_UNAUTHORISED_SA_NAME,
)
from utilities.certificates_utils import create_ca_bundle_file
from utilities.constants import Timeout
from utilities.resources.evalhub import EvalHub

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="class")
def evalhub_cr(
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[EvalHub, Any, Any]:
    """Create an EvalHub custom resource and wait for it to be ready."""
    with EvalHub(
        client=admin_client,
        name="evalhub",
        namespace=model_namespace.name,
        wait_for_resource=True,
    ) as evalhub:
        yield evalhub


@pytest.fixture(scope="class")
def evalhub_deployment(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    evalhub_cr: EvalHub,
) -> Deployment:
    """Wait for the EvalHub deployment to become available."""
    deployment = Deployment(
        client=admin_client,
        name=evalhub_cr.name,
        namespace=model_namespace.name,
    )
    deployment.wait_for_replicas(timeout=Timeout.TIMEOUT_5MIN)
    return deployment


@pytest.fixture(scope="class")
def evalhub_route(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    evalhub_deployment: Deployment,
) -> Route:
    """Get the Route created by the operator for the EvalHub service."""
    return Route(
        client=admin_client,
        name=evalhub_deployment.name,
        namespace=model_namespace.name,
        ensure_exists=True,
    )


@pytest.fixture(scope="class")
def evalhub_ca_bundle_file(
    admin_client: DynamicClient,
) -> str:
    """Create a CA bundle file for verifying the EvalHub route TLS certificate."""
    return create_ca_bundle_file(client=admin_client)


# Multi-Tenancy Fixtures


@pytest.fixture(scope="class")
def tenant_a_namespace(
    admin_client: DynamicClient,
) -> Generator[Namespace, Any, Any]:
    """Create tenant-a namespace with EvalHub tenant label."""
    with Namespace(
        client=admin_client,
        name=TENANT_A_NAME,
        label={EVALHUB_TENANT_LABEL: ""},
    ) as ns:
        ns.wait_for_status(status=Namespace.Status.ACTIVE, timeout=Timeout.TIMEOUT_2MIN)
        yield ns


@pytest.fixture(scope="class")
def tenant_b_namespace(
    admin_client: DynamicClient,
) -> Generator[Namespace, Any, Any]:
    """Create tenant-b namespace with EvalHub tenant label."""
    with Namespace(
        client=admin_client,
        name=TENANT_B_NAME,
        label={EVALHUB_TENANT_LABEL: ""},
    ) as ns:
        ns.wait_for_status(status=Namespace.Status.ACTIVE, timeout=Timeout.TIMEOUT_2MIN)
        yield ns


@pytest.fixture(scope="class")
def tenant_a_service_account(
    admin_client: DynamicClient,
    tenant_a_namespace: Namespace,
) -> Generator[ServiceAccount, Any, Any]:
    """Create service account for tenant-a."""
    with ServiceAccount(
        client=admin_client,
        name=TENANT_A_SA_NAME,
        namespace=tenant_a_namespace.name,
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def tenant_b_service_account(
    admin_client: DynamicClient,
    tenant_b_namespace: Namespace,
) -> Generator[ServiceAccount, Any, Any]:
    """Create service account for tenant-b."""
    with ServiceAccount(
        client=admin_client,
        name=TENANT_B_SA_NAME,
        namespace=tenant_b_namespace.name,
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def tenant_unauthorised_service_account(
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[ServiceAccount, Any, Any]:
    """Create service account without any EvalHub permissions."""
    with ServiceAccount(
        client=admin_client,
        name=TENANT_UNAUTHORISED_SA_NAME,
        namespace=model_namespace.name,
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def tenant_a_evaluator_role(
    admin_client: DynamicClient,
    tenant_a_namespace: Namespace,
) -> Generator[Role, Any, Any]:
    """Create evaluator role for tenant-a with permissions to manage evaluations."""
    with Role(
        client=admin_client,
        name=EVALHUB_EVALUATOR_ROLE,
        namespace=tenant_a_namespace.name,
        rules=[
            {
                "apiGroups": ["trustyai.opendatahub.io"],
                "resources": ["evaluations", "collections", "providers"],
                "verbs": ["get", "list", "create", "update", "delete"],
            },
            {
                "apiGroups": ["mlflow.kubeflow.org"],
                "resources": ["experiments"],
                "verbs": ["create", "get"],
            },
        ],
    ) as role:
        yield role


@pytest.fixture(scope="class")
def tenant_b_evaluator_role(
    admin_client: DynamicClient,
    tenant_b_namespace: Namespace,
) -> Generator[Role, Any, Any]:
    """Create evaluator role for tenant-b with permissions to manage evaluations."""
    with Role(
        client=admin_client,
        name=EVALHUB_EVALUATOR_ROLE,
        namespace=tenant_b_namespace.name,
        rules=[
            {
                "apiGroups": ["trustyai.opendatahub.io"],
                "resources": ["evaluations", "collections", "providers"],
                "verbs": ["get", "list", "create", "update", "delete"],
            },
            {
                "apiGroups": ["mlflow.kubeflow.org"],
                "resources": ["experiments"],
                "verbs": ["create", "get"],
            },
        ],
    ) as role:
        yield role


@pytest.fixture(scope="class")
def tenant_a_evaluator_role_binding(
    admin_client: DynamicClient,
    tenant_a_namespace: Namespace,
    tenant_a_evaluator_role: Role,
    tenant_a_service_account: ServiceAccount,
) -> Generator[RoleBinding, Any, Any]:
    """Bind evaluator role to tenant-a service account."""
    with RoleBinding(
        client=admin_client,
        name=f"{EVALHUB_EVALUATOR_ROLE}-binding",
        namespace=tenant_a_namespace.name,
        subjects_kind="ServiceAccount",
        subjects_name=tenant_a_service_account.name,
        role_ref_kind="Role",
        role_ref_name=tenant_a_evaluator_role.name,
    ) as rb:
        yield rb


@pytest.fixture(scope="class")
def tenant_b_evaluator_role_binding(
    admin_client: DynamicClient,
    tenant_b_namespace: Namespace,
    tenant_b_evaluator_role: Role,
    tenant_b_service_account: ServiceAccount,
) -> Generator[RoleBinding, Any, Any]:
    """Bind evaluator role to tenant-b service account."""
    with RoleBinding(
        client=admin_client,
        name=f"{EVALHUB_EVALUATOR_ROLE}-binding",
        namespace=tenant_b_namespace.name,
        subjects_kind="ServiceAccount",
        subjects_name=tenant_b_service_account.name,
        role_ref_kind="Role",
        role_ref_name=tenant_b_evaluator_role.name,
    ) as rb:
        yield rb


@pytest.fixture(scope="class")
def evalhub_providers_role_binding_tenant_a(
    admin_client: DynamicClient,
    tenant_a_namespace: Namespace,
    tenant_a_service_account: ServiceAccount,
) -> Generator[RoleBinding, Any, Any]:
    """Bind providers access cluster role to tenant-a service account."""
    with RoleBinding(
        client=admin_client,
        name="evalhub-providers-access",
        namespace=tenant_a_namespace.name,
        subjects_kind="ServiceAccount",
        subjects_name=tenant_a_service_account.name,
        role_ref_kind="ClusterRole",
        role_ref_name=EVALHUB_PROVIDERS_ACCESS_CLUSTER_ROLE,
    ) as rb:
        yield rb


@pytest.fixture(scope="class")
def evalhub_providers_role_binding_tenant_b(
    admin_client: DynamicClient,
    tenant_b_namespace: Namespace,
    tenant_b_service_account: ServiceAccount,
) -> Generator[RoleBinding, Any, Any]:
    """Bind providers access cluster role to tenant-b service account."""
    with RoleBinding(
        client=admin_client,
        name="evalhub-providers-access",
        namespace=tenant_b_namespace.name,
        subjects_kind="ServiceAccount",
        subjects_name=tenant_b_service_account.name,
        role_ref_kind="ClusterRole",
        role_ref_name=EVALHUB_PROVIDERS_ACCESS_CLUSTER_ROLE,
    ) as rb:
        yield rb


@pytest.fixture(scope="class")
def tenant_a_token(
    tenant_a_service_account: ServiceAccount,
    tenant_a_evaluator_role_binding: RoleBinding,
    evalhub_providers_role_binding_tenant_a: RoleBinding,
) -> str:
    """Generate bearer token for tenant-a service account (30 min validity)."""
    return run_command(
        shlex.split(
            f"oc create token -n {tenant_a_service_account.namespace} "
            f"{tenant_a_service_account.name} --duration=30m"
        )
    )[1].strip()


@pytest.fixture(scope="class")
def tenant_b_token(
    tenant_b_service_account: ServiceAccount,
    tenant_b_evaluator_role_binding: RoleBinding,
    evalhub_providers_role_binding_tenant_b: RoleBinding,
) -> str:
    """Generate bearer token for tenant-b service account (30 min validity)."""
    return run_command(
        shlex.split(
            f"oc create token -n {tenant_b_service_account.namespace} "
            f"{tenant_b_service_account.name} --duration=30m"
        )
    )[1].strip()


@pytest.fixture(scope="class")
def tenant_unauthorised_token(
    tenant_unauthorised_service_account: ServiceAccount,
) -> str:
    """Generate bearer token for unauthorised service account (30 min validity)."""
    return run_command(
        shlex.split(
            f"oc create token -n {tenant_unauthorised_service_account.namespace} "
            f"{tenant_unauthorised_service_account.name} --duration=30m"
        )
    )[1].strip()


@pytest.fixture(scope="class")
def multi_tenant_setup(
    tenant_a_namespace: Namespace,
    tenant_b_namespace: Namespace,
    tenant_a_token: str,
    tenant_b_token: str,
    tenant_unauthorised_token: str,
) -> dict[str, Any]:
    """Composite fixture providing all multi-tenancy setup."""
    return {
        "tenant_a": {
            "namespace": tenant_a_namespace,
            "token": tenant_a_token,
        },
        "tenant_b": {
            "namespace": tenant_b_namespace,
            "token": tenant_b_token,
        },
        "unauthorised": {
            "token": tenant_unauthorised_token,
        },
    }
