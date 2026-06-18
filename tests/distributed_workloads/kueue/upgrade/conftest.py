"""Fixtures and configuration for Kueue CRD migration upgrade tests.

This module provides pytest fixtures specific to Kueue upgrade testing,
following the established patterns from the opendatahub-tests framework.
"""

import json
from collections.abc import Generator
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.namespace import Namespace

from utilities.infra import create_ns
from utilities.kueue_utils import (
    wait_for_kueue_crds_available,
)
from utilities.kueue_utils_v1beta1 import (
    ClusterQueueV1Beta1,
    LocalQueueV1Beta1,
    ResourceFlavorV1Beta1,
    create_cluster_queue_v1beta1,
    create_local_queue_v1beta1,
    create_resource_flavor_v1beta1,
    create_test_namespace_selector,
    create_test_node_labels,
    create_test_resource_groups,
    create_test_tolerations,
)

LOGGER = structlog.get_logger(name=__name__)

# Test constants
KUEUE_MIGRATION_NAMESPACE = "kueue-crd-migration-test"
KUEUE_BASELINE_CM_NAME = "kueue-crd-migration-baseline"

# Resource names for migration testing
TEST_CLUSTER_QUEUE_NAME = "migration-test-cluster-queue"
TEST_LOCAL_QUEUE_NAME = "migration-test-local-queue"
TEST_CPU_FLAVOR_NAME = "migration-test-cpu-flavor"
TEST_GPU_FLAVOR_NAME = "migration-test-gpu-flavor"


@pytest.fixture(scope="session")
def kueue_migration_namespace(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    unprivileged_client: DynamicClient,
    teardown_resources: bool,
) -> Generator[Namespace, Any, Any]:
    """Namespace for Kueue CRD migration tests."""
    namespace_labels = {
        "kueue-migration-test": "enabled",
        "kueue.openshift.io/managed": "true",
    }

    ns = Namespace(
        client=unprivileged_client,
        name=KUEUE_MIGRATION_NAMESPACE,
    )

    if pytestconfig.option.post_upgrade:
        yield ns
        if teardown_resources:
            ns.client = admin_client
            ns.clean_up()
    else:
        with create_ns(
            admin_client=admin_client,
            unprivileged_client=unprivileged_client,
            name=KUEUE_MIGRATION_NAMESPACE,
            add_dashboard_label=True,
            teardown=teardown_resources,
            labels=namespace_labels,
        ) as ns:
            yield ns


@pytest.fixture(scope="session")
def kueue_crds_available(admin_client: DynamicClient, pytestconfig: pytest.Config) -> bool:
    """Ensure Kueue CRDs and controller are ready before testing."""
    api_version = "v1beta2" if pytestconfig.option.post_upgrade else "v1beta1"
    wait_for_kueue_crds_available(client=admin_client, api_version=api_version)
    LOGGER.info(f"Kueue {api_version} CRDs and controller confirmed ready")
    return True


@pytest.fixture(scope="session")
def test_cpu_resource_flavor_v1beta1(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    kueue_crds_available: bool,
    teardown_resources: bool,
) -> Generator[ResourceFlavorV1Beta1, Any, Any]:
    """CPU ResourceFlavor created with v1beta1 API for migration testing."""
    rf_kwargs = {
        "client": admin_client,
        "name": TEST_CPU_FLAVOR_NAME,
        "node_labels": create_test_node_labels(),
        "tolerations": create_test_tolerations(),
    }

    if pytestconfig.option.post_upgrade:
        rf = ResourceFlavorV1Beta1(**rf_kwargs)
        yield rf
        if teardown_resources:
            rf.clean_up()
    else:
        with create_resource_flavor_v1beta1(
            **rf_kwargs,
            teardown=teardown_resources,
        ) as rf:
            LOGGER.info(f"Created CPU ResourceFlavor '{rf.name}' using v1beta1 API")
            yield rf


@pytest.fixture(scope="session")
def test_gpu_resource_flavor_v1beta1(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    kueue_crds_available: bool,
    teardown_resources: bool,
) -> Generator[ResourceFlavorV1Beta1, Any, Any]:
    """GPU ResourceFlavor created with v1beta1 API for migration testing."""
    gpu_node_labels = {
        "node.kubernetes.io/instance-type": "gpu-node-type",
        "accelerator": "nvidia-t4",
        "topology.zone": "gpu-zone-a",
    }

    gpu_tolerations = [
        {
            "key": "nvidia.com/gpu",
            "operator": "Equal",
            "value": "true",
            "effect": "NoSchedule",
        }
    ]

    rf_kwargs = {
        "client": admin_client,
        "name": TEST_GPU_FLAVOR_NAME,
        "node_labels": gpu_node_labels,
        "tolerations": gpu_tolerations,
    }

    if pytestconfig.option.post_upgrade:
        rf = ResourceFlavorV1Beta1(**rf_kwargs)
        yield rf
        if teardown_resources:
            rf.clean_up()
    else:
        with create_resource_flavor_v1beta1(
            **rf_kwargs,
            teardown=teardown_resources,
        ) as rf:
            LOGGER.info(f"Created GPU ResourceFlavor '{rf.name}' using v1beta1 API")
            yield rf


@pytest.fixture(scope="session")
def test_cluster_queue_v1beta1(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    kueue_crds_available: bool,
    test_cpu_resource_flavor_v1beta1: ResourceFlavorV1Beta1,
    test_gpu_resource_flavor_v1beta1: ResourceFlavorV1Beta1,
    teardown_resources: bool,
) -> Generator[ClusterQueueV1Beta1, Any, Any]:
    """ClusterQueue created with v1beta1 API for migration testing."""
    resource_groups = create_test_resource_groups()

    cq_kwargs = {
        "client": admin_client,
        "name": TEST_CLUSTER_QUEUE_NAME,
        "resource_groups": resource_groups,
        "namespace_selector": create_test_namespace_selector(),
        "stop_policy": "Hold",  # Test policy conversion
        "cohort": "migration-test-cohort",  # Test cohort field
    }

    if pytestconfig.option.post_upgrade:
        cq = ClusterQueueV1Beta1(**cq_kwargs)
        yield cq
        if teardown_resources:
            cq.clean_up()
    else:
        with create_cluster_queue_v1beta1(
            **cq_kwargs,
            teardown=teardown_resources,
        ) as cq:
            LOGGER.info(f"Created ClusterQueue '{cq.name}' using v1beta1 API with stopPolicy=Hold")
            yield cq


@pytest.fixture(scope="session")
def test_local_queue_v1beta1(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    kueue_migration_namespace: Namespace,
    test_cluster_queue_v1beta1: ClusterQueueV1Beta1,
    teardown_resources: bool,
) -> Generator[LocalQueueV1Beta1, Any, Any]:
    """LocalQueue created with v1beta1 API for migration testing."""
    lq_kwargs = {
        "client": admin_client,
        "name": TEST_LOCAL_QUEUE_NAME,
        "cluster_queue": TEST_CLUSTER_QUEUE_NAME,
        "namespace": kueue_migration_namespace.name,
    }

    if pytestconfig.option.post_upgrade:
        yield LocalQueueV1Beta1(**lq_kwargs)
    else:
        with create_local_queue_v1beta1(
            **lq_kwargs,
            teardown=teardown_resources,
        ) as lq:
            LOGGER.info(f"Created LocalQueue '{lq.name}' using v1beta1 API")
            yield lq


@pytest.fixture(scope="session")
def capture_kueue_baseline(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    kueue_migration_namespace: Namespace,
    test_cpu_resource_flavor_v1beta1: ResourceFlavorV1Beta1,
    test_gpu_resource_flavor_v1beta1: ResourceFlavorV1Beta1,
    test_cluster_queue_v1beta1: ClusterQueueV1Beta1,
    test_local_queue_v1beta1: LocalQueueV1Beta1,
) -> None:
    """Capture Kueue resource baseline data before upgrade."""
    if pytestconfig.option.post_upgrade:
        return

    # Wait for resources to be fully created
    test_cpu_resource_flavor_v1beta1.wait()
    test_gpu_resource_flavor_v1beta1.wait()
    test_cluster_queue_v1beta1.wait()
    test_local_queue_v1beta1.wait()

    # Capture complete resource specifications from v1beta1 API
    baseline = {
        "cpu_resource_flavor": {
            "name": test_cpu_resource_flavor_v1beta1.name,
            "spec": test_cpu_resource_flavor_v1beta1.instance.spec,
            "generation": test_cpu_resource_flavor_v1beta1.instance.metadata.generation,
            "api_version": test_cpu_resource_flavor_v1beta1.instance.apiVersion,
        },
        "gpu_resource_flavor": {
            "name": test_gpu_resource_flavor_v1beta1.name,
            "spec": test_gpu_resource_flavor_v1beta1.instance.spec,
            "generation": test_gpu_resource_flavor_v1beta1.instance.metadata.generation,
            "api_version": test_gpu_resource_flavor_v1beta1.instance.apiVersion,
        },
        "cluster_queue": {
            "name": test_cluster_queue_v1beta1.name,
            "spec": test_cluster_queue_v1beta1.instance.spec,
            "generation": test_cluster_queue_v1beta1.instance.metadata.generation,
            "api_version": test_cluster_queue_v1beta1.instance.apiVersion,
            "status": getattr(test_cluster_queue_v1beta1.instance, "status", None),
        },
        "local_queue": {
            "name": test_local_queue_v1beta1.name,
            "spec": test_local_queue_v1beta1.instance.spec,
            "generation": test_local_queue_v1beta1.instance.metadata.generation,
            "api_version": test_local_queue_v1beta1.instance.apiVersion,
            "namespace": test_local_queue_v1beta1.namespace,
        },
    }

    ConfigMap(
        client=admin_client,
        name=KUEUE_BASELINE_CM_NAME,
        namespace=kueue_migration_namespace.name,
        data={"baseline": json.dumps(baseline, default=str, sort_keys=True)},
    ).deploy()

    LOGGER.info(f"Captured Kueue v1beta1 baseline: {len(baseline)} resources")


@pytest.fixture(scope="session")
def kueue_migration_baseline(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    kueue_migration_namespace: Namespace,
) -> dict[str, Any]:
    """Load pre-upgrade Kueue baseline from ConfigMap."""
    if not pytestconfig.option.post_upgrade:
        return {}

    cm = ConfigMap(
        client=admin_client,
        name=KUEUE_BASELINE_CM_NAME,
        namespace=kueue_migration_namespace.name,
    )

    assert cm.exists, (
        f"Kueue baseline ConfigMap '{KUEUE_BASELINE_CM_NAME}' not found. Ensure pre-upgrade tests ran successfully."
    )

    cm_data = cm.instance.data or {}
    raw = cm_data.get("baseline")
    assert raw, "Baseline ConfigMap has no 'baseline' key in data."

    baseline = json.loads(raw)
    LOGGER.info(f"Loaded Kueue migration baseline: {len(baseline)} resources")
    return baseline
