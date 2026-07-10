import uuid
from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.resource import ResourceEditor
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.service_account import ServiceAccount
from pytest_testconfig import py_config

from utilities.constants import TRUSTYAI_SERVICE_NAME
from utilities.infra import create_inference_token, create_ns, get_data_science_cluster

EVALHUB_TENANT_LABEL_KEY = "evalhub.trustyai.opendatahub.io/tenant"

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
def trustyai_operator_deployment(admin_client: DynamicClient) -> Deployment:
    return Deployment(
        client=admin_client,
        name=f"{TRUSTYAI_SERVICE_NAME}-operator-controller-manager",
        namespace=py_config["applications_namespace"],
        ensure_exists=True,
    )


@pytest.fixture(scope="class")
def patched_dsc_lmeval_allow_all(
    admin_client, trustyai_operator_deployment: Deployment
) -> Generator[DataScienceCluster]:
    """Enable LMEval PermitOnline and PermitCodeExecution flags in the Datascience cluster."""
    dsc = get_data_science_cluster(client=admin_client)
    with ResourceEditor(
        patches={
            dsc: {
                "spec": {
                    "components": {
                        "trustyai": {
                            "eval": {
                                "lmeval": {
                                    "permitCodeExecution": "allow",
                                    "permitOnline": "allow",
                                }
                            }
                        }
                    }
                }
            }
        }
    ):
        num_replicas: int = trustyai_operator_deployment.instance.spec.replicas
        trustyai_operator_deployment.scale_replicas(replica_count=0)
        trustyai_operator_deployment.scale_replicas(replica_count=num_replicas)
        trustyai_operator_deployment.wait_for_replicas()
        yield dsc


# ---------------------------------------------------------------------------
# EvalHub multi-tenancy fixtures (tenant namespaces, RBAC, tokens)
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
