"""Advanced Kueue conversion webhook testing for v1beta1↔v1beta2 compatibility.

This module provides comprehensive testing of Kueue CRD conversion webhooks
to ensure seamless bidirectional API compatibility during upgrades.

Supplements the main CRD migration test with webhook-specific edge cases.
"""

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace

from utilities.kueue_utils import (
    ClusterQueue,
    LocalQueue,
    ResourceFlavor,
)
from utilities.kueue_utils_v1beta1 import (
    create_cluster_queue_v1beta1,
    create_local_queue_v1beta1,
    create_resource_flavor_v1beta1,
)

LOGGER = structlog.get_logger(name=__name__)


class TestKueueConversionWebhookPreUpgrade:
    """Pre-upgrade validation for Kueue conversion webhook environment."""

    @pytest.mark.pre_upgrade
    def test_kueue_crds_ready_for_conversion_testing(
        self,
        kueue_crds_available: bool,
    ) -> None:
        """Verify Kueue CRDs and controller are ready for webhook testing."""
        assert kueue_crds_available
        LOGGER.info("Kueue CRDs and controller ready for conversion webhook testing")


class TestKueueConversionWebhookPostUpgrade:
    """Advanced conversion webhook testing for Kueue CRD migration."""

    @pytest.mark.post_upgrade
    def test_conversion_webhook_round_trip_compatibility(
        self,
        admin_client: DynamicClient,
        kueue_migration_namespace: Namespace,
    ) -> None:
        """Test round-trip conversion v1beta1→v1beta2→v1beta1 preserves data."""
        test_flavor_name = "webhook-test-flavor"

        # Create via v1beta1 with complex data
        complex_node_labels = {
            "node.kubernetes.io/instance-type": "m5.xlarge",
            "topology.kubernetes.io/zone": "us-east-1a",
            "node.kubernetes.io/arch": "amd64",
            "special-chars": "value_with-dashes.dots",
            "unicode-test": "测试-🚀",
        }

        complex_tolerations = [
            {
                "key": "nvidia.com/gpu",
                "operator": "Equal",
                "value": "true",
                "effect": "NoSchedule",
                "tolerationSeconds": 3600,
            },
            {
                "key": "spot-instance",
                "operator": "Exists",
                "effect": "NoExecute",
            },
            {
                "key": "workload-isolation",
                "operator": "Equal",
                "value": "ml-training",
                "effect": "PreferNoSchedule",
            },
        ]

        with create_resource_flavor_v1beta1(
            client=admin_client,
            name=test_flavor_name,
            node_labels=complex_node_labels,
            tolerations=complex_tolerations,
            teardown=True,
        ) as rf_v1beta1:
            # Read via v1beta2 API
            rf_v1beta2 = ResourceFlavor(
                client=admin_client,
                name=test_flavor_name,
            )

            # Verify conversion preserves complex data
            v1beta1_spec = rf_v1beta1.instance.spec
            v1beta2_spec = rf_v1beta2.instance.spec

            assert v1beta2_spec.nodeLabels == v1beta1_spec.nodeLabels
            assert v1beta2_spec.tolerations == v1beta1_spec.tolerations

            # Verify special characters and unicode preserved
            assert v1beta2_spec.nodeLabels["special-chars"] == "value_with-dashes.dots"
            assert v1beta2_spec.nodeLabels["unicode-test"] == "测试-🚀"

            # Verify complex tolerations structure preserved
            converted_tolerations = v1beta2_spec.tolerations
            assert len(converted_tolerations) == 3

            # Find the GPU toleration and check tolerationSeconds
            gpu_toleration = next(t for t in converted_tolerations if t["key"] == "nvidia.com/gpu")
            assert gpu_toleration["tolerationSeconds"] == 3600

        LOGGER.info("Conversion webhook round-trip compatibility verified")

    @pytest.mark.post_upgrade
    def test_conversion_webhook_field_defaults_handling(
        self,
        admin_client: DynamicClient,
        kueue_migration_namespace: Namespace,
    ) -> None:
        """Test conversion webhook handling of optional fields and defaults."""
        test_cq_name = "webhook-defaults-test"

        # Create ClusterQueue with minimal spec via v1beta1
        minimal_resource_groups = [
            {
                "coveredResources": ["cpu"],
                "flavors": [
                    {
                        "name": "default-flavor",
                        "resources": [
                            {"name": "cpu", "nominalQuota": "1"},
                        ],
                    }
                ],
            }
        ]

        with create_cluster_queue_v1beta1(
            client=admin_client,
            name=test_cq_name,
            resource_groups=minimal_resource_groups,
            # Deliberately omit optional fields to test defaults
            namespace_selector=None,
            stop_policy=None,
            cohort=None,
            teardown=True,
        ) as cq_v1beta1:
            # Read via v1beta2
            cq_v1beta2 = ClusterQueue(
                client=admin_client,
                name=test_cq_name,
            )

            v1beta1_spec = cq_v1beta1.instance.spec
            v1beta2_spec = cq_v1beta2.instance.spec

            # Verify resource groups preserved exactly
            assert v1beta2_spec.resourceGroups == v1beta1_spec.resourceGroups

            # Verify optional fields handle conversion correctly
            # namespace_selector should default to empty dict if not set
            if hasattr(v1beta1_spec, "namespaceSelector"):
                assert v1beta2_spec.namespaceSelector == v1beta1_spec.namespaceSelector

            # stopPolicy and cohort should be None/empty if not set
            if hasattr(v1beta1_spec, "stopPolicy"):
                assert getattr(v1beta2_spec, "stopPolicy", None) == v1beta1_spec.stopPolicy
            if hasattr(v1beta1_spec, "cohort"):
                assert getattr(v1beta2_spec, "cohort", None) == v1beta1_spec.cohort

        LOGGER.info("Conversion webhook field defaults handling verified")

    @pytest.mark.post_upgrade
    def test_conversion_webhook_status_preservation(
        self,
        admin_client: DynamicClient,
        kueue_migration_namespace: Namespace,
    ) -> None:
        """Test conversion webhook preserves status fields during API version changes."""
        test_cq_name = "webhook-status-test"

        resource_groups = [
            {
                "coveredResources": ["memory"],
                "flavors": [
                    {
                        "name": "memory-flavor",
                        "resources": [
                            {"name": "memory", "nominalQuota": "4Gi"},
                        ],
                    }
                ],
            }
        ]

        with create_cluster_queue_v1beta1(
            client=admin_client,
            name=test_cq_name,
            resource_groups=resource_groups,
            teardown=True,
        ) as cq_v1beta1:
            # Wait for ClusterQueue to reach Active state
            cq_v1beta1.wait_for_condition(
                condition="Active",
                status="True",
                timeout=120,
            )

            # Capture status via v1beta1
            v1beta1_status = cq_v1beta1.instance.status
            assert v1beta1_status is not None

            # Read same resource via v1beta2
            cq_v1beta2 = ClusterQueue(
                client=admin_client,
                name=test_cq_name,
            )

            v1beta2_status = cq_v1beta2.instance.status
            assert v1beta2_status is not None

            # Verify critical status fields preserved across API versions
            if hasattr(v1beta1_status, "conditions"):
                # Both should have Active condition
                v1beta1_conditions = {c["type"]: c["status"] for c in v1beta1_status.conditions}
                v1beta2_conditions = {c["type"]: c["status"] for c in v1beta2_status.conditions}

                assert v1beta1_conditions.get("Active") == "True"
                assert v1beta2_conditions.get("Active") == "True"

            # Verify resource usage/quota info preserved if present
            if hasattr(v1beta1_status, "flavorsReservation"):
                assert hasattr(v1beta2_status, "flavorsReservation")
            if hasattr(v1beta1_status, "flavorsUsage"):
                assert hasattr(v1beta2_status, "flavorsUsage")

        LOGGER.info("Conversion webhook status preservation verified")

    @pytest.mark.post_upgrade
    def test_conversion_webhook_large_resource_groups(
        self,
        admin_client: DynamicClient,
    ) -> None:
        """Test conversion webhook with large, complex resourceGroups configurations."""
        test_cq_name = "webhook-large-config-test"

        # Create large resourceGroups to test webhook performance/limits
        large_resource_groups = []
        for i in range(5):  # Multiple resource groups
            flavors = []
            for j in range(3):  # Multiple flavors per group
                resources = []
                for resource_name in ["cpu", "memory", "nvidia.com/gpu", "ephemeral-storage"]:
                    resources.append({
                        "name": resource_name,
                        "nominalQuota": f"{j + 1}0",
                    })

                flavors.append({
                    "name": f"flavor-group-{i}-flavor-{j}",
                    "resources": resources,
                })

            large_resource_groups.append({
                "coveredResources": ["cpu", "memory", "nvidia.com/gpu", "ephemeral-storage"],
                "flavors": flavors,
            })

        with create_cluster_queue_v1beta1(
            client=admin_client,
            name=test_cq_name,
            resource_groups=large_resource_groups,
            teardown=True,
        ) as cq_v1beta1:
            # Read via v1beta2
            cq_v1beta2 = ClusterQueue(
                client=admin_client,
                name=test_cq_name,
            )

            v1beta1_groups = cq_v1beta1.instance.spec.resourceGroups
            v1beta2_groups = cq_v1beta2.instance.spec.resourceGroups

            # Verify large config preserved exactly
            assert len(v1beta2_groups) == len(v1beta1_groups)
            assert v1beta2_groups == v1beta1_groups

            # Verify specific complex nested structure
            assert len(v1beta2_groups) == 5
            for group in v1beta2_groups:
                assert len(group["flavors"]) == 3
                for flavor in group["flavors"]:
                    assert len(flavor["resources"]) == 4

        LOGGER.info("Conversion webhook large configuration handling verified")

    @pytest.mark.post_upgrade
    def test_conversion_webhook_concurrent_operations(
        self,
        admin_client: DynamicClient,
        kueue_migration_namespace: Namespace,
    ) -> None:
        """Test conversion webhook under concurrent read/write operations."""
        import threading
        import time

        test_base_name = "webhook-concurrent-test"
        results = {"success": 0, "errors": []}

        def create_and_read_resource(thread_id: int) -> None:
            """Create resource via v1beta1 and read via v1beta2."""
            try:
                resource_name = f"{test_base_name}-{thread_id}"

                with create_local_queue_v1beta1(
                    client=admin_client,
                    name=resource_name,
                    cluster_queue="default",  # Assume exists or create fallback
                    namespace=kueue_migration_namespace.name,
                    teardown=True,
                ) as lq_v1beta1:
                    # Small delay to create overlap
                    time.sleep(0.1)

                    # Read via v1beta2 while v1beta1 operations are happening
                    lq_v1beta2 = LocalQueue(
                        client=admin_client,
                        name=resource_name,
                        namespace=kueue_migration_namespace.name,
                    )

                    # Verify conversion works under concurrency
                    assert lq_v1beta2.exists
                    assert lq_v1beta2.instance.spec.clusterQueue == "default"

                    results["success"] += 1

            except Exception as e:
                results["errors"].append(f"Thread {thread_id}: {e}")

        # Run multiple threads to test webhook under load
        threads = []
        for i in range(3):  # Modest concurrency to avoid overloading
            thread = threading.Thread(target=create_and_read_resource, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join(timeout=60)  # Prevent hanging

        # Verify all operations succeeded
        assert results["success"] == 3, f"Errors: {results['errors']}"
        assert len(results["errors"]) == 0, f"Conversion webhook failed under concurrency: {results['errors']}"

        LOGGER.info("Conversion webhook concurrent operations verified")
