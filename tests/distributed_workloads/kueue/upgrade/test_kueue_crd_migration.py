"""Kueue CRD v1beta1→v1beta2 migration test for RHOAI upgrades.

This module tests that Kueue resources created with v1beta1 API before upgrade
are successfully converted and accessible via v1beta2 API after upgrade.

Test addresses Gap G3 from RHOAIENG-63117 audit report.
"""

import structlog
import pytest
from typing import Any
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace

from utilities.kueue_utils import (
    ResourceFlavor,
    LocalQueue,
    ClusterQueue,
)
from utilities.kueue_utils_v1beta1 import (
    ResourceFlavorV1Beta1,
    LocalQueueV1Beta1,
    ClusterQueueV1Beta1,
)

# Import test constants from conftest
from .conftest import (
    TEST_CLUSTER_QUEUE_NAME,
    TEST_LOCAL_QUEUE_NAME,
    TEST_CPU_FLAVOR_NAME,
    TEST_GPU_FLAVOR_NAME,
)

LOGGER = structlog.get_logger(name=__name__)


class TestKueueCRDMigrationPreUpgrade:
    """Pre-upgrade tests: Create Kueue resources using v1beta1 API."""

    @pytest.mark.pre_upgrade
    def test_create_kueue_resources_v1beta1_pre_upgrade(
        self,
        admin_client: DynamicClient,
        kueue_migration_namespace: Namespace,
        test_cpu_resource_flavor_v1beta1: ResourceFlavorV1Beta1,
        test_gpu_resource_flavor_v1beta1: ResourceFlavorV1Beta1,
        test_cluster_queue_v1beta1: ClusterQueueV1Beta1,
        test_local_queue_v1beta1: LocalQueueV1Beta1,
        capture_kueue_baseline: None,
    ) -> None:
        """Verify Kueue resources are created successfully using v1beta1 API."""
        # Verify all resources exist and are using v1beta1 API
        assert test_cpu_resource_flavor_v1beta1.exists
        assert test_cpu_resource_flavor_v1beta1.instance.apiVersion == "kueue.x-k8s.io/v1beta1"

        assert test_gpu_resource_flavor_v1beta1.exists
        assert test_gpu_resource_flavor_v1beta1.instance.apiVersion == "kueue.x-k8s.io/v1beta1"

        assert test_cluster_queue_v1beta1.exists
        assert test_cluster_queue_v1beta1.instance.apiVersion == "kueue.x-k8s.io/v1beta1"

        assert test_local_queue_v1beta1.exists
        assert test_local_queue_v1beta1.instance.apiVersion == "kueue.x-k8s.io/v1beta1"

        # Verify ClusterQueue fields are properly set
        cq_spec = test_cluster_queue_v1beta1.instance.spec
        assert cq_spec.stopPolicy == "Hold"
        assert cq_spec.cohort == "migration-test-cohort"
        assert cq_spec.namespaceSelector.matchLabels["kueue-migration-test"] == "enabled"

        # Verify ResourceFlavor node labels and tolerations
        cpu_spec = test_cpu_resource_flavor_v1beta1.instance.spec
        # Access spec fields - handle both dict and attribute access patterns
        if hasattr(cpu_spec, "nodeLabels"):
            node_labels = cpu_spec.nodeLabels
        else:
            node_labels = cpu_spec.get("nodeLabels", {})
        if hasattr(cpu_spec, "tolerations"):
            tolerations = cpu_spec.tolerations
        else:
            tolerations = cpu_spec.get("tolerations", [])

        # Convert to regular dict if it's a special Kubernetes object
        if not isinstance(node_labels, dict):
            node_labels = dict(node_labels)

        assert "node.kubernetes.io/instance-type" in node_labels, (
            f"Expected key not found. Labels type: {type(node_labels)}, "
            f"Labels: {node_labels}, Keys: {list(node_labels.keys())}"
        )
        assert len(tolerations) > 0

        LOGGER.info("Successfully created all Kueue resources using v1beta1 API")

    @pytest.mark.pre_upgrade
    def test_kueue_resources_functional_v1beta1_pre_upgrade(
        self,
        admin_client: DynamicClient,
        kueue_migration_namespace: Namespace,
        test_cluster_queue_v1beta1: ClusterQueueV1Beta1,
        test_local_queue_v1beta1: LocalQueueV1Beta1,
    ) -> None:
        """Verify Kueue resources are functional before upgrade."""
        # ClusterQueue has stopPolicy=Hold, so it will have Active=False with Stopped reason
        # This is intentional for testing policy field preservation during migration
        # Verify the ClusterQueue has the expected Stopped condition
        test_cluster_queue_v1beta1.wait_for_condition(
            condition="Active",
            status="False",
            timeout=120,
        )

        # Verify LocalQueue references ClusterQueue correctly
        lq_spec = test_local_queue_v1beta1.instance.spec
        assert lq_spec.clusterQueue == TEST_CLUSTER_QUEUE_NAME

        LOGGER.info("Kueue resources are functional before upgrade")


class TestKueueCRDMigrationPostUpgrade:
    """Post-upgrade tests: Validate v1beta1→v1beta2 conversion and field preservation."""

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="kueue_resources_exist_v1beta2")
    def test_kueue_resources_accessible_v1beta2_post_upgrade(
        self,
        admin_client: DynamicClient,
        kueue_migration_namespace: Namespace,
        kueue_migration_baseline: dict[str, Any],
    ) -> None:
        """Verify Kueue resources are accessible via v1beta2 API after upgrade."""
        # Access resources using v1beta2 API classes
        cpu_flavor_v1beta2 = ResourceFlavor(
            client=admin_client,
            name=TEST_CPU_FLAVOR_NAME,
        )
        gpu_flavor_v1beta2 = ResourceFlavor(
            client=admin_client,
            name=TEST_GPU_FLAVOR_NAME,
        )
        cluster_queue_v1beta2 = ClusterQueue(
            client=admin_client,
            name=TEST_CLUSTER_QUEUE_NAME,
        )
        local_queue_v1beta2 = LocalQueue(
            client=admin_client,
            name=TEST_LOCAL_QUEUE_NAME,
            namespace=kueue_migration_namespace.name,
            cluster_queue=TEST_CLUSTER_QUEUE_NAME,
        )

        # Verify all resources exist via v1beta2 API
        assert cpu_flavor_v1beta2.exists
        assert gpu_flavor_v1beta2.exists
        assert cluster_queue_v1beta2.exists
        assert local_queue_v1beta2.exists

        # Verify API version has been converted
        assert cpu_flavor_v1beta2.instance.apiVersion == "kueue.x-k8s.io/v1beta2"
        assert gpu_flavor_v1beta2.instance.apiVersion == "kueue.x-k8s.io/v1beta2"
        assert cluster_queue_v1beta2.instance.apiVersion == "kueue.x-k8s.io/v1beta2"
        assert local_queue_v1beta2.instance.apiVersion == "kueue.x-k8s.io/v1beta2"

        LOGGER.info("All Kueue resources accessible via v1beta2 API after upgrade")

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["kueue_resources_exist_v1beta2"])
    def test_kueue_field_preservation_post_upgrade(
        self,
        admin_client: DynamicClient,
        kueue_migration_namespace: Namespace,
        kueue_migration_baseline: dict[str, Any],
    ) -> None:
        """Verify all fields are preserved during v1beta1→v1beta2 migration."""
        # Get current v1beta2 resources
        cpu_flavor_v1beta2 = ResourceFlavor(
            client=admin_client,
            name=TEST_CPU_FLAVOR_NAME,
        )
        gpu_flavor_v1beta2 = ResourceFlavor(
            client=admin_client,
            name=TEST_GPU_FLAVOR_NAME,
        )
        cluster_queue_v1beta2 = ClusterQueue(
            client=admin_client,
            name=TEST_CLUSTER_QUEUE_NAME,
        )
        local_queue_v1beta2 = LocalQueue(
            client=admin_client,
            name=TEST_LOCAL_QUEUE_NAME,
            namespace=kueue_migration_namespace.name,
            cluster_queue=TEST_CLUSTER_QUEUE_NAME,
        )

        # Compare CPU ResourceFlavor fields
        cpu_baseline = kueue_migration_baseline["cpu_resource_flavor"]["spec"]
        cpu_current = cpu_flavor_v1beta2.instance.spec

        assert cpu_current.nodeLabels == cpu_baseline["nodeLabels"]
        assert cpu_current.tolerations == cpu_baseline["tolerations"]

        # Compare GPU ResourceFlavor fields
        gpu_baseline = kueue_migration_baseline["gpu_resource_flavor"]["spec"]
        gpu_current = gpu_flavor_v1beta2.instance.spec

        assert gpu_current.nodeLabels == gpu_baseline["nodeLabels"]
        assert gpu_current.tolerations == gpu_baseline["tolerations"]

        # Compare ClusterQueue fields
        cq_baseline = kueue_migration_baseline["cluster_queue"]["spec"]
        cq_current = cluster_queue_v1beta2.instance.spec

        assert cq_current.stopPolicy == cq_baseline["stopPolicy"]
        assert cq_current.cohort == cq_baseline["cohort"]
        assert cq_current.namespaceSelector == cq_baseline["namespaceSelector"]
        assert cq_current.resourceGroups == cq_baseline["resourceGroups"]

        # Compare LocalQueue fields
        lq_baseline = kueue_migration_baseline["local_queue"]["spec"]
        lq_current = local_queue_v1beta2.instance.spec

        assert lq_current.clusterQueue == lq_baseline["clusterQueue"]

        LOGGER.info("All Kueue resource fields preserved during migration")

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["kueue_resources_exist_v1beta2"])
    def test_kueue_functional_behavior_post_upgrade(
        self,
        admin_client: DynamicClient,
        kueue_migration_namespace: Namespace,
    ) -> None:
        """Verify Kueue resources maintain functional behavior after upgrade."""
        cluster_queue_v1beta2 = ClusterQueue(
            client=admin_client,
            name=TEST_CLUSTER_QUEUE_NAME,
        )

        local_queue_v1beta2 = LocalQueue(
            client=admin_client,
            name=TEST_LOCAL_QUEUE_NAME,
            namespace=kueue_migration_namespace.name,
            cluster_queue=TEST_CLUSTER_QUEUE_NAME,
        )

        # Verify ClusterQueue is still Active after upgrade
        cluster_queue_v1beta2.wait_for_condition(
            condition="Active",
            status="True",
            timeout=120,
        )

        # Update ClusterQueue stopPolicy to None to test admission
        cluster_queue_v1beta2.update({
            "spec": {
                "stopPolicy": "None"
            }
        })

        # Verify update was successful
        cluster_queue_v1beta2.wait_for_condition(
            condition="Active",
            status="True",
            timeout=120,
        )

        updated_spec = cluster_queue_v1beta2.instance.spec
        assert updated_spec.stopPolicy == "None"

        LOGGER.info("Kueue resources maintain functional behavior after upgrade")

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["kueue_resources_exist_v1beta2"])
    def test_kueue_bidirectional_api_compatibility_post_upgrade(
        self,
        admin_client: DynamicClient,
        kueue_migration_namespace: Namespace,
    ) -> None:
        """Verify both v1beta1 and v1beta2 APIs work post-upgrade."""
        # Read same resource via both API versions
        cpu_flavor_v1beta1 = ResourceFlavorV1Beta1(
            client=admin_client,
            name=TEST_CPU_FLAVOR_NAME,
        )
        cpu_flavor_v1beta2 = ResourceFlavor(
            client=admin_client,
            name=TEST_CPU_FLAVOR_NAME,
        )

        # Both should exist and have same underlying data
        assert cpu_flavor_v1beta1.exists
        assert cpu_flavor_v1beta2.exists

        # Compare spec content (conversion webhook should handle translation)
        v1beta1_spec = cpu_flavor_v1beta1.instance.spec
        v1beta2_spec = cpu_flavor_v1beta2.instance.spec

        assert v1beta1_spec.nodeLabels == v1beta2_spec.nodeLabels
        assert v1beta1_spec.tolerations == v1beta2_spec.tolerations

        # Test same for ClusterQueue
        cq_v1beta1 = ClusterQueueV1Beta1(
            client=admin_client,
            name=TEST_CLUSTER_QUEUE_NAME,
        )
        cq_v1beta2 = ClusterQueue(
            client=admin_client,
            name=TEST_CLUSTER_QUEUE_NAME,
        )

        assert cq_v1beta1.exists
        assert cq_v1beta2.exists

        v1beta1_cq_spec = cq_v1beta1.instance.spec
        v1beta2_cq_spec = cq_v1beta2.instance.spec

        assert v1beta1_cq_spec.stopPolicy == v1beta2_cq_spec.stopPolicy
        assert v1beta1_cq_spec.cohort == v1beta2_cq_spec.cohort

        LOGGER.info("Bidirectional API compatibility verified post-upgrade")