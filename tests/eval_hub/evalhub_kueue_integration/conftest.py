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
) -> Generator[Namespace, Any, Any]:
    """Create a test namespace with Kueue management labels.

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
