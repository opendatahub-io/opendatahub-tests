"""
Integration test for Kueue and Notebook admission control.
Tests that Notebook CRs can be managed by Kueue queue system.
"""

import time
from datetime import datetime, timezone
import pytest
from simple_logger.logger import get_logger
from kubernetes.dynamic import DynamicClient
from ocp_resources.pod import Pod
from ocp_resources.namespace import Namespace
from ocp_resources.notebook import Notebook
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.resource import ResourceEditor
from utilities.kueue_utils import (
    check_gated_pods_and_running_pods,
    get_queue_resource_usage,
    verify_queue_tracks_workload,
)
from utilities.constants import Timeout

LOGGER = get_logger(name=__name__)

pytestmark = [
    pytest.mark.kueue,
    pytest.mark.smoke,
]

# Test constants
NAMESPACE_NAME = "kueue-notebook-test"
LOCAL_QUEUE_NAME = "notebook-local-queue"
CLUSTER_QUEUE_NAME = "notebook-cluster-queue"
RESOURCE_FLAVOR_NAME = "notebook-flavor"
CPU_QUOTA = "4"
MEMORY_QUOTA = "8Gi"
NOTEBOOK_NAME = "test-kueue-notebook"

# Constants for resource starvation test
LOW_RESOURCE_NAMESPACE_NAME = "kueue-low-resource-test"
LOW_RESOURCE_LOCAL_QUEUE_NAME = "low-resource-local-queue"
LOW_RESOURCE_CLUSTER_QUEUE_NAME = "low-resource-cluster-queue"
LOW_RESOURCE_FLAVOR_NAME = "low-resource-flavor"
LOW_CPU_QUOTA = "100m"  # Very low CPU quota - 100m
LOW_MEMORY_QUOTA = "64Mi"  # Very low memory quota
HIGH_DEMAND_NOTEBOOK_NAME = "high-demand-notebook"


@pytest.mark.parametrize(
    "patched_kueue_manager_config, kueue_enabled_notebook_namespace, kueue_notebook_persistent_volume_claim, "
    "default_notebook, kueue_notebook_cluster_queue, kueue_notebook_resource_flavor, kueue_notebook_local_queue",
    [
        pytest.param(
            {},  # Uses default patching behavior for ConfigMap
            {"name": NAMESPACE_NAME, "add-kueue-label": True},
            {"name": NOTEBOOK_NAME},
            {
                "namespace": NAMESPACE_NAME,
                "name": NOTEBOOK_NAME,
                "labels": {"kueue.x-k8s.io/queue-name": LOCAL_QUEUE_NAME},
            },
            {
                "name": CLUSTER_QUEUE_NAME,
                "resource_flavor_name": RESOURCE_FLAVOR_NAME,
                "cpu_quota": CPU_QUOTA,
                "memory_quota": MEMORY_QUOTA,
                "namespace_selector": {"matchLabels": {"kubernetes.io/metadata.name": NAMESPACE_NAME}},
            },
            {"name": RESOURCE_FLAVOR_NAME},
            {"name": LOCAL_QUEUE_NAME, "cluster_queue": CLUSTER_QUEUE_NAME},
            id="normal_resources",
        ),
        pytest.param(
            {},  # Uses default patching behavior for ConfigMap
            {"name": LOW_RESOURCE_NAMESPACE_NAME, "add-kueue-label": True},
            {"name": HIGH_DEMAND_NOTEBOOK_NAME},
            {
                "namespace": LOW_RESOURCE_NAMESPACE_NAME,
                "name": HIGH_DEMAND_NOTEBOOK_NAME,
                "labels": {"kueue.x-k8s.io/queue-name": LOW_RESOURCE_LOCAL_QUEUE_NAME},
                # Request resources that exceed the queue limits
                "cpu_request": "1000m",  # 1 CPU (exceeds 100m limit)
                "memory_request": "1Gi",  # 1 GB (exceeds 64Mi limit)
            },
            {
                "name": LOW_RESOURCE_CLUSTER_QUEUE_NAME,
                "resource_flavor_name": LOW_RESOURCE_FLAVOR_NAME,
                "cpu_quota": LOW_CPU_QUOTA,
                "memory_quota": LOW_MEMORY_QUOTA,
                "namespace_selector": {"matchLabels": {"kubernetes.io/metadata.name": LOW_RESOURCE_NAMESPACE_NAME}},
            },
            {"name": LOW_RESOURCE_FLAVOR_NAME},
            {
                "name": LOW_RESOURCE_LOCAL_QUEUE_NAME,
                "cluster_queue": LOW_RESOURCE_CLUSTER_QUEUE_NAME,
            },
            id="resource_starvation",
        ),
    ],
    indirect=True,
)
class TestKueueNotebookController:
    """Test Kueue integration with Notebook Controller

    PREREQUISITE: The patched_kueue_manager_config fixture runs FIRST and:
    1. Stores the original kueue-manager-config ConfigMap state
    2. Patches the ConfigMap with required frameworks and annotation
    3. Restarts kueue-controller-manager deployment to apply changes
    4. Runs all tests with the patched configuration
    5. CLEANUP: Restores original ConfigMap state and restarts deployment

    This ensures proper isolation and cleanup between test runs.
    """

    def test_kueue_notebook_admission_control(
        self,
        admin_client: DynamicClient,
        unprivileged_client: DynamicClient,
        patched_kueue_manager_config,
        kueue_enabled_notebook_namespace: Namespace,
        kueue_notebook_persistent_volume_claim: PersistentVolumeClaim,
        default_notebook: Notebook,
        kueue_notebook_resource_flavor,
        kueue_notebook_cluster_queue,
        kueue_notebook_local_queue,
    ):
        """
        Test that Kueue properly controls admission of Notebook workloads

        1. Create a Notebook CR with Kueue labels
        2. Verify the notebook pod is created and managed by Kueue
        3. Check that Kueue admission control is working (pod should be gated initially)
        4. Verify the notebook eventually becomes ready
        """

        # Skip this test for resource starvation scenario
        if default_notebook.name == HIGH_DEMAND_NOTEBOOK_NAME:
            pytest.skip("Skipping admission control test for resource starvation scenario")

        # Verify that the notebook was created with Kueue labels
        assert default_notebook.exists, "Notebook CR should be created successfully"
        notebook_labels = default_notebook.instance.metadata.labels or {}
        queue_name = notebook_labels.get("kueue.x-k8s.io/queue-name")
        assert queue_name is not None, (
            f"Notebook should have Kueue queue label. Available labels: {list(notebook_labels.keys())}"
        )
        assert queue_name == LOCAL_QUEUE_NAME, (
            f"Notebook should have correct queue name: {LOCAL_QUEUE_NAME}, got: {queue_name}"
        )

        # Find the notebook pod
        notebook_pod = Pod(
            client=unprivileged_client,
            namespace=default_notebook.namespace,
            name=f"{default_notebook.name}-0",
        )

        # Wait for the pod to be created
        notebook_pod.wait(timeout=Timeout.TIMEOUT_2MIN)

        # Check Kueue admission control behavior
        # The pod should initially be gated by Kueue, then allowed to run
        pod_labels = [
            f"app={default_notebook.name}",  # Notebook pods use the 'app' label
        ]

        # Verify that Kueue has admitted the workload and pod exists
        # The pod may be in various states: Running, Pending (ContainerCreating), or initially Gated
        assert notebook_pod.exists, f"Notebook pod {notebook_pod.name} should exist"

        # Check if the pod has Kueue management labels (proves Kueue is managing it)
        notebook_pod_labels = notebook_pod.instance.metadata.labels or {}
        assert notebook_pod_labels.get("kueue.x-k8s.io/managed") == "true", "Pod should be managed by Kueue"
        assert notebook_pod_labels.get("kueue.x-k8s.io/queue-name") == LOCAL_QUEUE_NAME, (
            "Pod should reference correct queue"
        )

        # Wait for the notebook pod to become ready
        notebook_pod.wait_for_condition(
            condition=Pod.Condition.READY, status=Pod.Condition.Status.TRUE, timeout=Timeout.TIMEOUT_5MIN
        )

        # Verify the pod is now running (not gated)
        running_pods_final, gated_pods_final = check_gated_pods_and_running_pods(
            labels=pod_labels, namespace=kueue_enabled_notebook_namespace.name, admin_client=admin_client
        )

        assert running_pods_final == 1, f"Expected exactly 1 running notebook pod, found {running_pods_final}"
        assert gated_pods_final == 0, f"Expected no gated notebook pods, found {gated_pods_final}"

        # ADDITIONAL CHECK: Verify Kueue is tracking exact resource usage
        # This proves that Kueue is actually managing the workload, not just adding labels
        LOGGER.info(f"Checking ClusterQueue {kueue_notebook_cluster_queue.name} resource usage...")
        cluster_queue_usage = get_queue_resource_usage(
            queue=kueue_notebook_cluster_queue, flavor_name=RESOURCE_FLAVOR_NAME
        )
        LOGGER.info(f"ClusterQueue usage: {cluster_queue_usage}")

        LOGGER.info(f"Checking LocalQueue {kueue_notebook_local_queue.name} resource usage...")
        local_queue_usage = get_queue_resource_usage(queue=kueue_notebook_local_queue, flavor_name=RESOURCE_FLAVOR_NAME)
        LOGGER.info(f"LocalQueue usage: {local_queue_usage}")

        # Determine expected resource usage based on total pod requests
        # For normal_resources scenario:
        # - Main notebook container: cpu="1", memory="1Gi"
        # - OAuth proxy sidecar: cpu="100m", memory="64Mi"
        # - Total: cpu="1100m", memory="1088Mi"
        expected_cpu = "1100m"
        expected_memory = "1088Mi"

        # Verify that either ClusterQueue or LocalQueue shows the exact expected resource usage
        cluster_tracks_exact = verify_queue_tracks_workload(
            queue=kueue_notebook_cluster_queue,
            flavor_name=RESOURCE_FLAVOR_NAME,
            expected_cpu=expected_cpu,
            expected_memory=expected_memory,
        )
        local_tracks_exact = verify_queue_tracks_workload(
            queue=kueue_notebook_local_queue,
            flavor_name=RESOURCE_FLAVOR_NAME,
            expected_cpu=expected_cpu,
            expected_memory=expected_memory,
        )

        assert cluster_tracks_exact or local_tracks_exact, (
            f"Either ClusterQueue or LocalQueue should show exact resource usage matching notebook requests "
            f"(CPU: {expected_cpu}, Memory: {expected_memory}). "
            f"ClusterQueue usage: {cluster_queue_usage}, LocalQueue usage: {local_queue_usage}"
        )

    def test_kueue_notebook_resource_constraints(
        self,
        admin_client: DynamicClient,
        unprivileged_client: DynamicClient,
        patched_kueue_manager_config,
        kueue_enabled_notebook_namespace: Namespace,
        kueue_notebook_persistent_volume_claim: PersistentVolumeClaim,
        default_notebook: Notebook,
        kueue_notebook_resource_flavor,
        kueue_notebook_cluster_queue,
        kueue_notebook_local_queue,
    ):
        """
        Test that Kueue enforces resource constraints for notebooks

        1. Verify the notebook's resource requests are within the queue limits
        2. Check that Kueue properly tracks resource usage
        """

        # Skip this test for resource starvation scenario
        if default_notebook.name == HIGH_DEMAND_NOTEBOOK_NAME:
            pytest.skip("Skipping resource constraints test for resource starvation scenario")

        # Verify notebook exists and has proper configuration
        assert default_notebook.exists, "Notebook CR should exist"

        # Get the notebook pod
        notebook_pod = Pod(
            client=unprivileged_client,
            namespace=default_notebook.namespace,
            name=f"{default_notebook.name}-0",
        )

        # Wait for pod to be created and ready
        notebook_pod.wait(timeout=Timeout.TIMEOUT_2MIN)
        notebook_pod.wait_for_condition(
            condition=Pod.Condition.READY, status=Pod.Condition.Status.TRUE, timeout=Timeout.TIMEOUT_5MIN
        )

        # Verify resource requests are properly set
        pod_spec = notebook_pod.instance.spec
        containers = pod_spec.containers

        # Find the main notebook container (should be the first one)
        notebook_container = None
        for container in containers:
            if container.name == default_notebook.name:
                notebook_container = container
                break

        assert notebook_container is not None, "Notebook container should be found in pod spec"

        # Check resource requests exist
        resources = notebook_container.resources
        assert resources is not None, "Container should have resource specifications"
        assert resources.requests is not None, "Container should have resource requests"

        # Verify CPU and memory requests are within queue limits
        cpu_request = resources.requests.get("cpu", "0")
        memory_request = resources.requests.get("memory", "0")

        # Convert CPU request to numeric value for comparison
        if cpu_request.endswith("m"):
            cpu_value = int(cpu_request[:-1]) / 1000
        else:
            cpu_value = float(cpu_request)

        assert cpu_value <= float(CPU_QUOTA), f"CPU request {cpu_value} should not exceed queue quota {CPU_QUOTA}"

        # Memory requests should be reasonable (basic validation)
        assert memory_request != "0", "Memory request should be specified"

    def test_kueue_notebook_resource_starvation(
        self,
        admin_client: DynamicClient,
        unprivileged_client: DynamicClient,
        patched_kueue_manager_config,  # Ensures this runs first
        kueue_enabled_notebook_namespace: Namespace,
        kueue_notebook_persistent_volume_claim: PersistentVolumeClaim,
        default_notebook: Notebook,
        kueue_notebook_resource_flavor,
        kueue_notebook_cluster_queue,
        kueue_notebook_local_queue,
    ):
        """
        Test that Kueue properly gates Notebook workloads when there are insufficient resources

        This test creates:
        1. A queue with very low resource limits (100m CPU, 64Mi memory)
        2. A notebook that requests high resources (1 CPU, 1Gi memory)
        3. Verifies that the notebook pod remains in SchedulingGated state due to resource starvation
        """

        # Skip this test for normal resource scenario
        if default_notebook.name != HIGH_DEMAND_NOTEBOOK_NAME:
            pytest.skip("Skipping resource starvation test for normal resource scenario")

        # Verify that the notebook was created with Kueue labels
        assert default_notebook.exists, "High-demand Notebook CR should be created successfully"
        notebook_labels = default_notebook.instance.metadata.labels or {}
        queue_name = notebook_labels.get("kueue.x-k8s.io/queue-name")
        assert queue_name is not None, (
            f"Notebook should have Kueue queue label. Available labels: {list(notebook_labels.keys())}"
        )
        assert queue_name == LOW_RESOURCE_LOCAL_QUEUE_NAME, (
            f"Notebook should have correct queue name: {LOW_RESOURCE_LOCAL_QUEUE_NAME}, got: {queue_name}"
        )

        # Check that the notebook pod is created but gated due to insufficient resources
        notebook_pod = Pod(
            client=unprivileged_client,
            namespace=default_notebook.namespace,
            name=f"{default_notebook.name}-0",
        )

        # Wait for the pod to be created (but it should remain gated)
        notebook_pod.wait(timeout=Timeout.TIMEOUT_2MIN)
        assert notebook_pod.exists, f"Notebook pod {notebook_pod.name} should exist"

        # Verify that Kueue has applied management labels to the pod
        notebook_pod_labels = notebook_pod.instance.metadata.labels or {}
        assert notebook_pod_labels.get("kueue.x-k8s.io/managed") == "true", "Pod should be managed by Kueue"
        assert notebook_pod_labels.get("kueue.x-k8s.io/queue-name") == LOW_RESOURCE_LOCAL_QUEUE_NAME, (
            "Pod should reference correct queue"
        )

        # Define pod labels for the check function
        pod_labels = [f"app={default_notebook.name}"]

        # Wait for pod to be properly gated (check multiple times to ensure stable state)
        for _ in range(10):
            notebook_pod.get()  # Refresh pod state
            if notebook_pod.instance.status.phase == "Pending":
                conditions = notebook_pod.instance.status.conditions or []
                if any(c.type == "PodScheduled" and c.reason == "SchedulingGated" for c in conditions):
                    break
            time.sleep(1)  # noqa: FCN001

        # Check that the pod is gated due to insufficient resources
        running_pods, gated_pods = check_gated_pods_and_running_pods(
            labels=pod_labels, namespace=kueue_enabled_notebook_namespace.name, admin_client=admin_client
        )

        # The pod should be gated (not running) due to resource constraints
        assert gated_pods == 1, f"Expected exactly 1 gated notebook pod due to resource starvation, found {gated_pods}"
        assert running_pods == 0, f"Expected no running notebook pods due to resource constraints, found {running_pods}"

        # Verify the pod phase and conditions
        notebook_pod.get()  # Refresh pod state
        pod_status = notebook_pod.instance.status
        assert pod_status.phase == "Pending", f"Pod should be in Pending state, got: {pod_status.phase}"

        # Check for SchedulingGated condition
        scheduling_gated = False
        if hasattr(pod_status, "conditions") and pod_status.conditions:
            for condition in pod_status.conditions:
                if (
                    condition.type == "PodScheduled"
                    and condition.status == "False"
                    and condition.reason == "SchedulingGated"
                ):
                    scheduling_gated = True
                    break

        assert scheduling_gated, "Pod should have SchedulingGated condition due to insufficient resources"

        LOGGER.info(
            f"SUCCESS: Notebook pod {notebook_pod.name} is properly gated due to insufficient resources "
            f"in queue {LOW_RESOURCE_LOCAL_QUEUE_NAME}"
        )

    def test_kueue_notebook_stop_start_workbench(
        self,
        admin_client: DynamicClient,
        unprivileged_client: DynamicClient,
        patched_kueue_manager_config,  # Ensures this runs first
        kueue_enabled_notebook_namespace: Namespace,
        kueue_notebook_persistent_volume_claim: PersistentVolumeClaim,
        default_notebook: Notebook,
        kueue_notebook_resource_flavor,
        kueue_notebook_cluster_queue,
        kueue_notebook_local_queue,
    ):
        """
        Test that Kueue properly handles workbench stop/start lifecycle operations

        This test:
        1. Waits for a notebook to be running with sufficient resources
        2. Stops the workbench by adding kubeflow-resource-stopped annotation
        3. Verifies the pod is terminated
        4. Starts the workbench by removing the annotation
        5. Verifies the pod is recreated and running again
        """

        # Skip this test for resource starvation scenario (needs running notebook)
        if default_notebook.name == HIGH_DEMAND_NOTEBOOK_NAME:
            pytest.skip("Skipping stop/start test for resource starvation scenario")

        # Verify that the notebook was created with Kueue labels
        assert default_notebook.exists, "Notebook CR should be created successfully"
        notebook_labels = default_notebook.instance.metadata.labels or {}
        queue_name = notebook_labels.get("kueue.x-k8s.io/queue-name")
        assert queue_name is not None, (
            f"Notebook should have Kueue queue label. Available labels: {list(notebook_labels.keys())}"
        )
        assert queue_name == LOCAL_QUEUE_NAME, (
            f"Notebook should have correct queue name: {LOCAL_QUEUE_NAME}, got: {queue_name}"
        )

        # Wait for the notebook pod to be created and running
        notebook_pod = Pod(
            client=unprivileged_client,
            namespace=default_notebook.namespace,
            name=f"{default_notebook.name}-0",
        )

        # Wait for the pod to be created and become ready
        notebook_pod.wait(timeout=Timeout.TIMEOUT_2MIN)
        assert notebook_pod.exists, f"Notebook pod {notebook_pod.name} should exist"

        # Verify that Kueue has applied management labels to the pod
        notebook_pod_labels = notebook_pod.instance.metadata.labels or {}
        assert notebook_pod_labels.get("kueue.x-k8s.io/managed") == "true", "Pod should be managed by Kueue"
        assert notebook_pod_labels.get("kueue.x-k8s.io/queue-name") == LOCAL_QUEUE_NAME, (
            "Pod should reference correct queue"
        )

        # Wait for the notebook pod to become ready (initial startup)
        notebook_pod.wait_for_condition(
            condition=Pod.Condition.READY, status=Pod.Condition.Status.TRUE, timeout=Timeout.TIMEOUT_5MIN
        )

        LOGGER.info(f"SUCCESS: Notebook pod {notebook_pod.name} is initially running and ready")

        # Verify Kueue is tracking exact resource usage for the running workload
        LOGGER.info("Checking queue resource usage before stopping workbench...")
        initial_cluster_usage = get_queue_resource_usage(
            queue=kueue_notebook_cluster_queue, flavor_name=RESOURCE_FLAVOR_NAME
        )
        initial_local_usage = get_queue_resource_usage(
            queue=kueue_notebook_local_queue, flavor_name=RESOURCE_FLAVOR_NAME
        )

        LOGGER.info(f"Initial ClusterQueue usage: {initial_cluster_usage}")
        LOGGER.info(f"Initial LocalQueue usage: {initial_local_usage}")

        # For normal_resources scenario:
        # - Main notebook container: cpu="1", memory="1Gi"
        # - OAuth proxy sidecar: cpu="100m", memory="64Mi"
        # - Total: cpu="1100m", memory="1088Mi"
        expected_cpu = "1100m"
        expected_memory = "1088Mi"

        # Verify that Kueue is actively tracking the exact expected resource usage
        cluster_tracks_exact = verify_queue_tracks_workload(
            queue=kueue_notebook_cluster_queue,
            flavor_name=RESOURCE_FLAVOR_NAME,
            expected_cpu=expected_cpu,
            expected_memory=expected_memory,
        )
        local_tracks_exact = verify_queue_tracks_workload(
            queue=kueue_notebook_local_queue,
            flavor_name=RESOURCE_FLAVOR_NAME,
            expected_cpu=expected_cpu,
            expected_memory=expected_memory,
        )

        assert cluster_tracks_exact or local_tracks_exact, (
            f"Kueue should be tracking exact resource usage matching notebook requests "
            f"(CPU: {expected_cpu}, Memory: {expected_memory}). "
            f"ClusterQueue usage: {initial_cluster_usage}, LocalQueue usage: {initial_local_usage}"
        )

        # STEP 1: Stop the workbench by adding the kubeflow-resource-stopped annotation
        stop_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: FCN001

        # Get current notebook annotations
        default_notebook.get()  # Refresh to get latest state
        current_annotations = default_notebook.instance.metadata.annotations or {}

        # Add the stop annotation
        updated_annotations = {**current_annotations, "kubeflow-resource-stopped": stop_timestamp}

        # Patch the notebook with the stop annotation
        stop_patch = {"metadata": {"annotations": updated_annotations}}

        with ResourceEditor(patches={default_notebook: stop_patch}):
            LOGGER.info(f"Applied kubeflow-resource-stopped annotation with timestamp: {stop_timestamp}")

            # Wait for the pod to be terminated (it should be deleted)
            notebook_pod.wait_deleted(timeout=Timeout.TIMEOUT_2MIN)

            LOGGER.info(f"SUCCESS: Notebook pod {notebook_pod.name} was terminated after stop annotation")

            # Verify resource usage goes to zero after stopping workbench
            LOGGER.info("Checking queue resource usage after stopping workbench...")
            stopped_cluster_usage = get_queue_resource_usage(
                queue=kueue_notebook_cluster_queue, flavor_name=RESOURCE_FLAVOR_NAME
            )
            stopped_local_usage = get_queue_resource_usage(
                queue=kueue_notebook_local_queue, flavor_name=RESOURCE_FLAVOR_NAME
            )

            LOGGER.info(f"Stopped ClusterQueue usage: {stopped_cluster_usage}")
            LOGGER.info(f"Stopped LocalQueue usage: {stopped_local_usage}")

            # Verify that resource usage is now zero (no workloads running)
            cluster_zero_usage = verify_queue_tracks_workload(
                queue=kueue_notebook_cluster_queue,
                flavor_name=RESOURCE_FLAVOR_NAME,
                expected_cpu="0",
                expected_memory="0",
            )
            local_zero_usage = verify_queue_tracks_workload(
                queue=kueue_notebook_local_queue,
                flavor_name=RESOURCE_FLAVOR_NAME,
                expected_cpu="0",
                expected_memory="0",
            )

            # At least one queue should show zero usage (the workload is stopped)
            assert cluster_zero_usage or local_zero_usage, (
                "After stopping workbench, either ClusterQueue or LocalQueue should show zero resource usage. "
                f"ClusterQueue usage: {stopped_cluster_usage}, LocalQueue usage: {stopped_local_usage}"
            )

        # STEP 2: Start the workbench by removing the kubeflow-resource-stopped annotation
        time.sleep(5)  # Brief pause before restart  # noqa: FCN001

        # Get current notebook annotations again
        default_notebook.get()  # Refresh to get latest state
        current_annotations = default_notebook.instance.metadata.annotations or {}

        # Remove the stop annotation
        restart_annotations = {k: v for k, v in current_annotations.items() if k != "kubeflow-resource-stopped"}

        # Patch the notebook to remove the stop annotation
        restart_patch = {"metadata": {"annotations": restart_annotations}}

        with ResourceEditor(patches={default_notebook: restart_patch}):
            LOGGER.info("Removed kubeflow-resource-stopped annotation to restart workbench")

            # Wait for the new pod to be created (pod name should be the same)
            new_notebook_pod = Pod(
                client=unprivileged_client,
                namespace=default_notebook.namespace,
                name=f"{default_notebook.name}-0",
            )

            new_notebook_pod.wait(timeout=Timeout.TIMEOUT_2MIN)
            assert new_notebook_pod.exists, f"New notebook pod {new_notebook_pod.name} should be created after restart"

            # Verify that Kueue management labels are still present on the new pod
            new_notebook_pod_labels = new_notebook_pod.instance.metadata.labels or {}
            assert new_notebook_pod_labels.get("kueue.x-k8s.io/managed") == "true", (
                "Restarted pod should still be managed by Kueue"
            )
            assert new_notebook_pod_labels.get("kueue.x-k8s.io/queue-name") == LOCAL_QUEUE_NAME, (
                "Restarted pod should still reference correct queue"
            )

            # Wait for the new pod to become ready
            new_notebook_pod.wait_for_condition(
                condition=Pod.Condition.READY, status=Pod.Condition.Status.TRUE, timeout=Timeout.TIMEOUT_5MIN
            )

            LOGGER.info(f"SUCCESS: Notebook pod {new_notebook_pod.name} was successfully restarted and is ready again")

            # Final verification: Ensure Kueue is tracking exact resource usage for the restarted workload
            LOGGER.info("Checking queue resource usage after restarting workbench...")
            restarted_cluster_usage = get_queue_resource_usage(
                queue=kueue_notebook_cluster_queue, flavor_name=RESOURCE_FLAVOR_NAME
            )
            restarted_local_usage = get_queue_resource_usage(
                queue=kueue_notebook_local_queue, flavor_name=RESOURCE_FLAVOR_NAME
            )

            LOGGER.info(f"Restarted ClusterQueue usage: {restarted_cluster_usage}")
            LOGGER.info(f"Restarted LocalQueue usage: {restarted_local_usage}")

            # Verify that Kueue is again tracking the exact expected resource usage
            cluster_tracks_exact_restart = verify_queue_tracks_workload(
                queue=kueue_notebook_cluster_queue,
                flavor_name=RESOURCE_FLAVOR_NAME,
                expected_cpu=expected_cpu,
                expected_memory=expected_memory,
            )
            local_tracks_exact_restart = verify_queue_tracks_workload(
                queue=kueue_notebook_local_queue,
                flavor_name=RESOURCE_FLAVOR_NAME,
                expected_cpu=expected_cpu,
                expected_memory=expected_memory,
            )

            assert cluster_tracks_exact_restart or local_tracks_exact_restart, (
                f"Kueue should be tracking exact resource usage for the restarted workload "
                f"(CPU: {expected_cpu}, Memory: {expected_memory}). "
                f"ClusterQueue usage: {restarted_cluster_usage}, LocalQueue usage: {restarted_local_usage}"
            )


@pytest.mark.parametrize(
    "patched_kueue_manager_config",
    [
        pytest.param(
            {},  # Uses default patching behavior
        )
    ],
    indirect=True,
)
def test_kueue_config_frameworks_enabled(
    admin_client: DynamicClient,
    patched_kueue_manager_config,
):
    """
    Test that the kueue-manager-config has been properly patched
    to include pod and statefulset frameworks, and that the
    kueue-controller-manager deployment was restarted.

    NOTE: This test runs independently and will restore the original
    configuration after completion. If patched_kueue_manager_config is None,
    it means we're using Red Hat build of Kueue operator and this test is skipped.
    """
    import yaml
    from pytest_testconfig import config as py_config
    from ocp_resources.deployment import Deployment

    # Skip test if ConfigMap patching was skipped (Red Hat build of Kueue operator scenario)
    if patched_kueue_manager_config is None:
        pytest.skip("Skipping kueue-manager-config test for Red Hat build of Kueue operator scenario")
        return

    # Refresh the ConfigMap instance to get the latest state
    patched_kueue_manager_config.get()

    # Verify the ConfigMap was patched correctly
    config_data = patched_kueue_manager_config.instance.data
    assert config_data is not None, "ConfigMap should have data"

    config_yaml = config_data.get("controller_manager_config.yaml", "")
    assert config_yaml, "ConfigMap should contain controller_manager_config.yaml data"

    # Verify the annotation was set correctly
    annotations = patched_kueue_manager_config.instance.metadata.annotations or {}
    managed_value = annotations.get("opendatahub.io/managed")
    assert managed_value is not None, (
        f"ConfigMap should have opendatahub.io/managed annotation. Current annotations: {list(annotations.keys())}"
    )
    assert managed_value == "false", (
        f"ConfigMap should have opendatahub.io/managed set to 'false', got: {managed_value}"
    )

    # Parse the configuration
    config_dict = yaml.safe_load(config_yaml)
    assert config_dict is not None, "Configuration should be valid YAML"

    # Verify integrations section exists
    assert "integrations" in config_dict, "Configuration should have integrations section"
    assert "frameworks" in config_dict["integrations"], "Integrations should have frameworks section"

    frameworks = config_dict["integrations"]["frameworks"]
    assert isinstance(frameworks, list), "Frameworks should be a list"
    assert "pod" in frameworks, "Frameworks should include 'pod'"
    assert "statefulset" in frameworks, "Frameworks should include 'statefulset'"

    # Verify the kueue-controller-manager deployment is running and ready
    kueue_deployment = Deployment(
        client=admin_client,
        name="kueue-controller-manager",
        namespace=py_config["applications_namespace"],
        ensure_exists=True,
    )

    # Verify deployment is ready by checking status
    kueue_deployment.wait_for_condition(
        condition=kueue_deployment.Condition.AVAILABLE, status=kueue_deployment.Condition.Status.TRUE, timeout=60
    )

    # If the deployment is AVAILABLE, it means it's working correctly
    LOGGER.info("SUCCESS: kueue-controller-manager deployment is ready and available")
