from collections.abc import Generator
from typing import Any

import pytest
import structlog
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace

from tests.eval_hub.evalhub_kueue_integration.constants import (
    CLUSTER_QUEUE_NAME,
    EVALHUB_TENANT_LABEL,
    KUEUE_MANAGED_LABEL,
    LOCAL_QUEUE_NAME,
    RESOURCE_FLAVOR_NAME,
)
from utilities.infra import create_ns
from utilities.kueue_utils import (
    ClusterQueue,
    LocalQueue,
    ResourceFlavor,
    WorkloadPriorityClass,
    create_cluster_queue,
    create_local_queue,
    create_resource_flavor,
    create_workload_priority_class,
    wait_for_kueue_crds_available,
)

LOGGER = structlog.get_logger(name=__name__)


def _provision_evalhub_tenant_resources(
    admin_client: DynamicClient,
    tenant_namespace: str,
    evalhub_namespace: str,
) -> None:
    """Provision tenant namespace with EvalHub resources needed for job execution.

    Copies ServiceAccount, RoleBindings, and ConfigMap from EvalHub namespace to tenant namespace.
    """
    from ocp_resources.config_map import ConfigMap
    from ocp_resources.role import Role
    from ocp_resources.role_binding import RoleBinding
    from ocp_resources.service_account import ServiceAccount

    # ServiceAccount name pattern: evalhub-{namespace}-job
    sa_name = f"evalhub-{evalhub_namespace}-job"

    # Get source ServiceAccount from EvalHub namespace
    source_sa = ServiceAccount(client=admin_client, name=sa_name, namespace=evalhub_namespace)
    if not source_sa.exists:
        LOGGER.warning(
            "ServiceAccount not found in EvalHub namespace, tenant jobs may fail",
            sa_name=sa_name,
            evalhub_namespace=evalhub_namespace,
        )
        return

    # Create ServiceAccount in tenant namespace
    tenant_sa = ServiceAccount(
        client=admin_client,
        name=sa_name,
        namespace=tenant_namespace,
        teardown=False,  # Don't auto-delete when fixture cleans up
    )
    tenant_sa.create()
    LOGGER.info("Created ServiceAccount in tenant namespace", sa_name=sa_name, namespace=tenant_namespace)

    # Copy RoleBindings
    for rb_name in [f"{sa_name}-config-rb", f"{sa_name}-mlflow-job-rb", f"{sa_name}-access-rb"]:
        try:
            source_rb = RoleBinding(client=admin_client, name=rb_name, namespace=evalhub_namespace)
            if not source_rb.exists:
                LOGGER.warning("RoleBinding not found, skipping", rb_name=rb_name, evalhub_namespace=evalhub_namespace)
                continue

            # Get the source RoleBinding spec
            source_dict = source_rb.instance.to_dict()

            # Create new RoleBinding with copied roleRef and subjects
            tenant_rb = RoleBinding(
                client=admin_client,
                name=rb_name,
                namespace=tenant_namespace,
                teardown=False,
            )

            # Build the roleBinding resource body
            rb_body = {
                "roleRef": source_dict.get("roleRef"),
                "subjects": source_dict.get("subjects", []),
            }

            # Update subject namespace references
            for subject in rb_body["subjects"]:
                if subject.get("name") == sa_name:
                    subject["namespace"] = tenant_namespace

            # Create with the body
            tenant_rb.res = tenant_rb.client.resources.get(
                api_version="rbac.authorization.k8s.io/v1", kind="RoleBinding"
            )
            tenant_rb.res.create(
                body={
                    "apiVersion": "rbac.authorization.k8s.io/v1",
                    "kind": "RoleBinding",
                    "metadata": {"name": rb_name, "namespace": tenant_namespace},
                    **rb_body,
                },
                namespace=tenant_namespace,
            )
            LOGGER.info("Created RoleBinding in tenant namespace", rb_name=rb_name, namespace=tenant_namespace)

            # If RoleBinding references a Role (not ClusterRole), copy it too
            role_ref = source_dict.get("roleRef", {})
            if role_ref.get("kind") == "Role":
                role_name = role_ref.get("name")
                try:
                    source_role = Role(client=admin_client, name=role_name, namespace=evalhub_namespace)
                    if source_role.exists:
                        # Get the source Role
                        role_dict = source_role.instance.to_dict()

                        # Create Role in tenant namespace
                        tenant_role = Role(
                            client=admin_client,
                            name=role_name,
                            namespace=tenant_namespace,
                            teardown=False,
                        )
                        tenant_role.res = tenant_role.client.resources.get(
                            api_version="rbac.authorization.k8s.io/v1", kind="Role"
                        )
                        tenant_role.res.create(
                            body={
                                "apiVersion": "rbac.authorization.k8s.io/v1",
                                "kind": "Role",
                                "metadata": {"name": role_name, "namespace": tenant_namespace},
                                "rules": role_dict.get("rules", []),
                            },
                            namespace=tenant_namespace,
                        )
                        LOGGER.info("Created Role in tenant namespace", role_name=role_name, namespace=tenant_namespace)
                except Exception as role_err:
                    LOGGER.warning("Failed to copy Role", role_name=role_name, error=str(role_err))

        except Exception as e:
            LOGGER.warning("Failed to copy RoleBinding", rb_name=rb_name, error=str(e))

    # Copy evalhub-service-ca ConfigMap
    try:
        source_cm = ConfigMap(client=admin_client, name="evalhub-service-ca", namespace=evalhub_namespace)
        if not source_cm.exists:
            LOGGER.warning(
                "ConfigMap not found, skipping", cm_name="evalhub-service-ca", evalhub_namespace=evalhub_namespace
            )
            return

        # Get the source ConfigMap data
        source_dict = source_cm.instance.to_dict()

        # Create new ConfigMap with copied data
        tenant_cm = ConfigMap(
            client=admin_client,
            name="evalhub-service-ca",
            namespace=tenant_namespace,
            teardown=False,
            data=source_dict.get("data", {}),
        )
        tenant_cm.create()
        LOGGER.info("Created ConfigMap in tenant namespace", cm_name="evalhub-service-ca", namespace=tenant_namespace)
    except Exception as e:
        LOGGER.warning("Failed to copy ConfigMap", cm_name="evalhub-service-ca", error=str(e))


def _kueue_resource_groups(
    flavor_name: str,
    cpu_quota: int | str,
    memory_quota: str,
) -> list[dict[str, Any]]:
    return [
        {
            "coveredResources": ["cpu", "memory"],
            "flavors": [
                {
                    "name": flavor_name,
                    "resources": [
                        {"name": "cpu", "nominalQuota": cpu_quota},
                        {"name": "memory", "nominalQuota": memory_quota},
                    ],
                }
            ],
        }
    ]


@pytest.fixture(scope="session")
def ensure_kueue_available(admin_client: DynamicClient) -> None:
    """Ensure Kueue CRDs and controller pods are available."""
    wait_for_kueue_crds_available(client=admin_client)


@pytest.fixture(scope="class")
def eval_test_namespace(
    request: FixtureRequest,
    admin_client: DynamicClient,
    evalhub_namespace: str,
) -> Generator[Namespace, Any, Any]:
    """Create a test namespace with Kueue management labels and EvalHub resources.

    Parametrize with: {"name": "ns-name"} or {"name": "ns-name", "labels": {...}}
    """
    ns_name = request.param.get("name", "evalhub-kueue-test")
    extra_labels = request.param.get("labels", {})

    labels = {
        KUEUE_MANAGED_LABEL: "true",
        EVALHUB_TENANT_LABEL: "true",
        **extra_labels,
    }

    with create_ns(
        name=ns_name,
        admin_client=admin_client,
        labels=labels,
    ) as ns:
        # Provision tenant namespace with EvalHub resources
        _provision_evalhub_tenant_resources(
            admin_client=admin_client,
            tenant_namespace=ns_name,
            evalhub_namespace=evalhub_namespace,
        )
        yield ns


@pytest.fixture(scope="class")
def eval_resource_flavor(
    request: FixtureRequest,
    admin_client: DynamicClient,
    ensure_kueue_available: None,
) -> Generator[ResourceFlavor, Any, Any]:
    """Create a ResourceFlavor for eval tests.

    Parametrize with: {"name": "flavor-name"}
    """
    name = request.param.get("name", RESOURCE_FLAVOR_NAME)
    with create_resource_flavor(client=admin_client, name=name) as rf:
        yield rf


@pytest.fixture(scope="class")
def eval_cluster_queue(
    request: FixtureRequest,
    admin_client: DynamicClient,
    ensure_kueue_available: None,
) -> Generator[ClusterQueue, Any, Any]:
    """Create a ClusterQueue with configurable quotas.

    Parametrize with: {
        "name": "cq-name",
        "resource_flavor_name": "flavor",
        "cpu_quota": 2,
        "memory_quota": "8Gi",
        "namespace_selector": {...},  # optional
        "preemption": {...},  # optional
    }
    """
    name = request.param.get("name", CLUSTER_QUEUE_NAME)
    flavor = request.param.get("resource_flavor_name", RESOURCE_FLAVOR_NAME)
    cpu = request.param.get("cpu_quota", 2)
    memory = request.param.get("memory_quota", "8Gi")
    ns_selector = request.param.get("namespace_selector", {})
    preemption = request.param.get("preemption")

    with create_cluster_queue(
        client=admin_client,
        name=name,
        resource_groups=_kueue_resource_groups(flavor, cpu, memory),
        namespace_selector=ns_selector,
    ) as cq:
        if preemption:
            cq_dict = cq.instance.to_dict()
            cq_dict["spec"]["preemption"] = preemption
            cq.update(cq_dict)
        yield cq


@pytest.fixture(scope="class")
def eval_local_queue(
    request: FixtureRequest,
    admin_client: DynamicClient,
    eval_test_namespace: Namespace,
    ensure_kueue_available: None,
) -> Generator[LocalQueue, Any, Any]:
    """Create a LocalQueue in the test namespace.

    Parametrize with: {"name": "lq-name", "cluster_queue": "cq-name"}
    """
    name = request.param.get("name", LOCAL_QUEUE_NAME)
    cq_name = request.param.get("cluster_queue", CLUSTER_QUEUE_NAME)

    with create_local_queue(
        client=admin_client,
        name=name,
        cluster_queue=cq_name,
        namespace=eval_test_namespace.name,
    ) as lq:
        yield lq


@pytest.fixture(scope="class")
def eval_workload_priority_class(
    request: FixtureRequest,
    admin_client: DynamicClient,
    ensure_kueue_available: None,
) -> Generator[WorkloadPriorityClass, Any, Any]:
    """Create a WorkloadPriorityClass.

    Parametrize with: {"name": "priority-name", "value": 1000}
    """
    name = request.param["name"]
    value = request.param["value"]
    with create_workload_priority_class(client=admin_client, name=name, value=value) as wpc:
        yield wpc
