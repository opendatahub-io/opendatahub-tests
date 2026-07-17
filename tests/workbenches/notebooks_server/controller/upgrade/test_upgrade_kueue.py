"""Upgrade tests for Kueue-managed notebook workbenches.

Verifies that notebooks scheduled through Kueue queues survive a platform
upgrade with their management labels, queue infrastructure, and lifecycle
state preserved.

Pre-upgrade: creates kueue infrastructure and notebooks, captures baseline.
Post-upgrade: verifies everything survived, tests new notebook creation.
"""

from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.notebook import Notebook
from ocp_resources.pod import Pod

from tests.workbenches.notebooks_server.controller.upgrade.kueue_constants import (
    UPGRADE_KUEUE_CLUSTER_QUEUE_NAME,
    UPGRADE_KUEUE_LOCAL_QUEUE_NAME,
)
from tests.workbenches.notebooks_server.controller.utils import (
    KUBEFLOW_STOPPED_ANNOTATION,
    StatefulSet,
)
from utilities.kueue_utils import (
    KUEUE_CLUSTER_QUEUE_LABEL,
    KUEUE_LOCAL_QUEUE_LABEL,
    KUEUE_MANAGED_LABEL,
    KUEUE_QUEUE_NAME_LABEL,
    ClusterQueue,
    LocalQueue,
    ResourceFlavor,
    Workload,
)


@pytest.mark.usefixtures("kueue_statefulset_framework_check", "capture_kueue_baseline")
class TestPreUpgradeKueueNotebook:
    """Verify a kueue-managed notebook is running before the platform upgrade.

    Steps:
        1. Create kueue infrastructure (ResourceFlavor, ClusterQueue, LocalQueue).
        2. Create a Notebook CR with kueue.x-k8s.io/queue-name label.
        3. Wait for the pod to reach Ready state with Kueue management labels.
        4. Stop a second notebook for the stopped-notebook upgrade scenario.
        5. Capture baseline to a ConfigMap for post-upgrade comparison.
    """

    @pytest.mark.pre_upgrade
    def test_kueue_notebook_running_before_upgrade(
        self,
        upgrade_kueue_notebook_pod: Pod,
    ) -> None:
        """Given a kueue-managed Notebook CR is created before upgrade,
        When the notebook controller and Kueue admit the workload,
        Then the notebook pod should exist and be in Ready state.
        """

    @pytest.mark.pre_upgrade
    def test_kueue_labels_present_before_upgrade(
        self,
        upgrade_kueue_notebook_pod: Pod,
    ) -> None:
        """Given a kueue-managed notebook pod is running before upgrade,
        When Kueue admits the workload,
        Then the pod should have the full set of Kueue scheduling labels.
        """
        pod_labels = upgrade_kueue_notebook_pod.instance.metadata.labels or {}
        assert pod_labels.get(KUEUE_MANAGED_LABEL) == "true", (
            f"Notebook pod should have '{KUEUE_MANAGED_LABEL}=true' label before upgrade. "
            f"Labels: {list(pod_labels.keys())}"
        )
        assert pod_labels.get(KUEUE_QUEUE_NAME_LABEL) == UPGRADE_KUEUE_LOCAL_QUEUE_NAME, (
            f"Notebook pod should have '{KUEUE_QUEUE_NAME_LABEL}={UPGRADE_KUEUE_LOCAL_QUEUE_NAME}' "
            f"label. Got: {pod_labels.get(KUEUE_QUEUE_NAME_LABEL)}"
        )
        assert pod_labels.get(KUEUE_CLUSTER_QUEUE_LABEL) == UPGRADE_KUEUE_CLUSTER_QUEUE_NAME, (
            f"Notebook pod should have '{KUEUE_CLUSTER_QUEUE_LABEL}={UPGRADE_KUEUE_CLUSTER_QUEUE_NAME}' "
            f"label. Got: {pod_labels.get(KUEUE_CLUSTER_QUEUE_LABEL)}"
        )
        assert pod_labels.get(KUEUE_LOCAL_QUEUE_LABEL) == UPGRADE_KUEUE_LOCAL_QUEUE_NAME, (
            f"Notebook pod should have '{KUEUE_LOCAL_QUEUE_LABEL}={UPGRADE_KUEUE_LOCAL_QUEUE_NAME}' "
            f"label. Got: {pod_labels.get(KUEUE_LOCAL_QUEUE_LABEL)}"
        )

    @pytest.mark.pre_upgrade
    def test_kueue_queue_infrastructure_exists_before_upgrade(
        self,
        upgrade_kueue_resource_flavor: ResourceFlavor,
        upgrade_kueue_cluster_queue: ClusterQueue,
        upgrade_kueue_local_queue: LocalQueue,
    ) -> None:
        """Given kueue queue infrastructure is created before upgrade,
        When the Kueue controller reconciles,
        Then ResourceFlavor, ClusterQueue, and LocalQueue should all exist.
        """
        assert upgrade_kueue_resource_flavor.exists, (
            f"ResourceFlavor '{upgrade_kueue_resource_flavor.name}' should exist before upgrade"
        )
        assert upgrade_kueue_cluster_queue.exists, (
            f"ClusterQueue '{upgrade_kueue_cluster_queue.name}' should exist before upgrade"
        )
        assert upgrade_kueue_local_queue.exists, (
            f"LocalQueue '{upgrade_kueue_local_queue.name}' should exist before upgrade"
        )


@pytest.mark.usefixtures("kueue_statefulset_framework_check")
class TestPostUpgradeKueueNotebook:
    """Verify the kueue-managed notebook survived the platform upgrade.

    Steps:
        1. Verify the notebook pod still exists and was not restarted.
        2. Verify Kueue management labels are preserved on the pod.
        3. Verify the Notebook CR was not modified.
        4. Verify queue infrastructure (ClusterQueue, LocalQueue, ResourceFlavor) survived.
    """

    @pytest.mark.post_upgrade
    def test_kueue_notebook_not_restarted_after_upgrade(
        self,
        upgrade_kueue_notebook_pod: Pod,
        upgrade_kueue_baseline: dict[str, Any],
    ) -> None:
        """Given a kueue-managed notebook was running before upgrade,
        When the upgrade completes,
        Then the pod's creationTimestamp should match the pre-upgrade baseline.
        """
        assert upgrade_kueue_notebook_pod.exists, (
            f"Kueue notebook pod '{upgrade_kueue_notebook_pod.name}' no longer exists after upgrade"
        )

        current_timestamp = upgrade_kueue_notebook_pod.instance.metadata.creationTimestamp
        saved_timestamp = upgrade_kueue_baseline["pod_creation_timestamp"]

        assert current_timestamp == saved_timestamp, (
            f"Kueue notebook pod was restarted during upgrade. "
            f"Pre-upgrade creationTimestamp: {saved_timestamp}, "
            f"post-upgrade creationTimestamp: {current_timestamp}"
        )

    @pytest.mark.post_upgrade
    def test_kueue_management_labels_preserved_after_upgrade(
        self,
        upgrade_kueue_notebook_pod: Pod,
        upgrade_kueue_baseline: dict[str, Any],
    ) -> None:
        """Given a kueue-managed notebook pod had scheduling labels before upgrade,
        When the upgrade completes,
        Then all Kueue labels (managed, queue-name, cluster-queue, local-queue) should be unchanged.
        """
        assert upgrade_kueue_notebook_pod.exists, (
            f"Kueue notebook pod '{upgrade_kueue_notebook_pod.name}' no longer exists after upgrade"
        )

        pod_labels = upgrade_kueue_notebook_pod.instance.metadata.labels or {}

        saved_managed = upgrade_kueue_baseline["pod_kueue_managed_label"]
        current_managed = pod_labels.get(KUEUE_MANAGED_LABEL, "")
        assert current_managed == saved_managed, (
            f"Kueue managed label changed during upgrade. "
            f"Pre-upgrade: '{saved_managed}', post-upgrade: '{current_managed}'"
        )

        saved_queue_name = upgrade_kueue_baseline["pod_queue_name_label"]
        current_queue_name = pod_labels.get(KUEUE_QUEUE_NAME_LABEL, "")
        assert current_queue_name == saved_queue_name, (
            f"Kueue queue-name label changed during upgrade. "
            f"Pre-upgrade: '{saved_queue_name}', post-upgrade: '{current_queue_name}'"
        )

        saved_cluster_queue = upgrade_kueue_baseline["pod_cluster_queue_label"]
        current_cluster_queue = pod_labels.get(KUEUE_CLUSTER_QUEUE_LABEL, "")
        assert current_cluster_queue == saved_cluster_queue, (
            f"Kueue cluster-queue label changed during upgrade. "
            f"Pre-upgrade: '{saved_cluster_queue}', post-upgrade: '{current_cluster_queue}'"
        )

        saved_local_queue = upgrade_kueue_baseline["pod_local_queue_label"]
        current_local_queue = pod_labels.get(KUEUE_LOCAL_QUEUE_LABEL, "")
        assert current_local_queue == saved_local_queue, (
            f"Kueue local-queue label changed during upgrade. "
            f"Pre-upgrade: '{saved_local_queue}', post-upgrade: '{current_local_queue}'"
        )

    @pytest.mark.post_upgrade
    def test_kueue_notebook_cr_not_modified_after_upgrade(
        self,
        upgrade_kueue_notebook: Notebook,
        upgrade_kueue_baseline: dict[str, Any],
    ) -> None:
        """Given a kueue Notebook CR existed before upgrade,
        When the upgrade completes,
        Then the Notebook CR generation should be unchanged.
        """
        current_generation = upgrade_kueue_notebook.instance.metadata.generation
        saved_generation = upgrade_kueue_baseline["notebook_generation"]

        assert current_generation == saved_generation, (
            f"Kueue Notebook CR was modified during upgrade. "
            f"Pre-upgrade generation: {saved_generation}, "
            f"post-upgrade generation: {current_generation}"
        )

    @pytest.mark.post_upgrade
    def test_kueue_queue_infrastructure_survives_upgrade(
        self,
        upgrade_kueue_resource_flavor: ResourceFlavor,
        upgrade_kueue_cluster_queue: ClusterQueue,
        upgrade_kueue_local_queue: LocalQueue,
    ) -> None:
        """Given kueue queue infrastructure existed before upgrade,
        When the upgrade completes,
        Then ResourceFlavor, ClusterQueue, and LocalQueue should still exist.
        """
        assert upgrade_kueue_resource_flavor.exists, (
            f"ResourceFlavor '{upgrade_kueue_resource_flavor.name}' no longer exists after upgrade"
        )
        assert upgrade_kueue_cluster_queue.exists, (
            f"ClusterQueue '{upgrade_kueue_cluster_queue.name}' no longer exists after upgrade"
        )
        assert upgrade_kueue_local_queue.exists, (
            f"LocalQueue '{upgrade_kueue_local_queue.name}' no longer exists after upgrade"
        )

    @pytest.mark.post_upgrade
    def test_kueue_statefulset_healthy_after_upgrade(
        self,
        upgrade_kueue_notebook_statefulset: StatefulSet,
    ) -> None:
        """Given a kueue notebook StatefulSet existed before upgrade,
        When the upgrade completes,
        Then readyReplicas should equal spec.replicas.
        """
        assert upgrade_kueue_notebook_statefulset.exists, (
            f"StatefulSet '{upgrade_kueue_notebook_statefulset.name}' no longer exists after upgrade"
        )
        sts = upgrade_kueue_notebook_statefulset.instance
        expected_replicas = sts.spec.replicas
        ready_replicas = sts.status.readyReplicas or 0

        assert ready_replicas == expected_replicas, (
            f"Kueue notebook StatefulSet has {ready_replicas} ready replicas, "
            f"expected {expected_replicas} after upgrade"
        )

    @pytest.mark.post_upgrade
    def test_kueue_workload_admitted_after_upgrade(
        self,
        admin_client: DynamicClient,
        upgrade_kueue_notebook: Notebook,
        upgrade_kueue_namespace: Namespace,
    ) -> None:
        """Given a kueue-managed notebook had an admitted Workload before upgrade,
        When the upgrade completes,
        Then the Workload object should still exist and remain in Admitted state
            with the correct ClusterQueue assignment.
        """
        workloads = list(Workload.get(client=admin_client, namespace=upgrade_kueue_namespace.name))
        notebook_workloads = [wl for wl in workloads if upgrade_kueue_notebook.name in wl.name]
        assert notebook_workloads, (
            f"Kueue Workload for notebook '{upgrade_kueue_notebook.name}' no longer exists after upgrade"
        )

        workload = notebook_workloads[0]
        conditions = workload.instance.status.get("conditions", [])
        is_admitted = any(c["type"] == "Admitted" and c["status"] == "True" for c in conditions)
        assert is_admitted, (
            f"Workload '{workload.name}' should still be Admitted after upgrade. Conditions: {conditions}"
        )

        admission = workload.instance.status.get("admission", {})
        assert admission.get("clusterQueue") == UPGRADE_KUEUE_CLUSTER_QUEUE_NAME, (
            f"Workload should still be admitted to ClusterQueue "
            f"'{UPGRADE_KUEUE_CLUSTER_QUEUE_NAME}' after upgrade, "
            f"got: '{admission.get('clusterQueue')}'"
        )


@pytest.mark.usefixtures("upgrade_kueue_stopped_pre_upgrade_shutdown")
class TestPostUpgradeKueueStopped:
    """Verify a stopped kueue-managed notebook remains stopped after upgrade.

    Steps:
        1. Verify the stop annotation is still present and unchanged.
        2. Verify the StatefulSet still has replicas=0.
        3. Verify no pod exists for the stopped notebook.
    """

    @pytest.mark.pre_upgrade
    def test_kueue_stopped_notebook_has_zero_replicas(
        self,
        upgrade_kueue_stopped_notebook_statefulset: StatefulSet,
    ) -> None:
        """Given a kueue notebook was stopped via kubeflow-resource-stopped annotation,
        When the controller reconciles the StatefulSet,
        Then the StatefulSet should have 0 replicas.
        """
        replicas = upgrade_kueue_stopped_notebook_statefulset.instance.spec.replicas
        assert replicas == 0, (
            f"StatefulSet '{upgrade_kueue_stopped_notebook_statefulset.name}' has {replicas} replicas, expected 0"
        )

    @pytest.mark.pre_upgrade
    def test_kueue_stopped_notebook_pod_absent(
        self,
        upgrade_kueue_stopped_notebook: Notebook,
        unprivileged_client: DynamicClient,
    ) -> None:
        """Given a kueue notebook was stopped,
        When the pod terminates,
        Then no pod should exist for the stopped notebook.
        """
        notebook_pod = Pod(
            client=unprivileged_client,
            namespace=upgrade_kueue_stopped_notebook.namespace,
            name=f"{upgrade_kueue_stopped_notebook.name}-0",
        )
        assert not notebook_pod.exists, f"Pod '{notebook_pod.name}' still exists for stopped kueue notebook"

    @pytest.mark.post_upgrade
    def test_kueue_stopped_annotation_preserved_after_upgrade(
        self,
        upgrade_kueue_stopped_notebook: Notebook,
    ) -> None:
        """Given a kueue notebook was stopped before upgrade,
        When the upgrade completes,
        Then the kubeflow-resource-stopped annotation should still be present.
        """
        stop_annotation = upgrade_kueue_stopped_notebook.instance.metadata.annotations.get(KUBEFLOW_STOPPED_ANNOTATION)
        assert stop_annotation is not None, (
            f"Kueue notebook '{upgrade_kueue_stopped_notebook.name}' lost "
            f"'{KUBEFLOW_STOPPED_ANNOTATION}' annotation after upgrade"
        )

    @pytest.mark.post_upgrade
    def test_kueue_stopped_annotation_value_unchanged_after_upgrade(
        self,
        upgrade_kueue_stopped_notebook: Notebook,
        upgrade_kueue_baseline: dict[str, Any],
    ) -> None:
        """Given a kueue notebook was stopped with a specific timestamp before upgrade,
        When the upgrade completes,
        Then the annotation value should be unchanged.
        """
        current_value = upgrade_kueue_stopped_notebook.instance.metadata.annotations.get(KUBEFLOW_STOPPED_ANNOTATION)
        saved_value = upgrade_kueue_baseline["stopped_annotation_value"]

        assert current_value == saved_value, (
            f"Stop annotation value changed during upgrade. "
            f"Pre-upgrade: '{saved_value}', post-upgrade: '{current_value}'"
        )

    @pytest.mark.post_upgrade
    def test_kueue_stopped_notebook_still_has_zero_replicas(
        self,
        upgrade_kueue_stopped_notebook_statefulset: StatefulSet,
    ) -> None:
        """Given a kueue notebook was stopped before upgrade,
        When the upgrade completes,
        Then the StatefulSet should still have 0 replicas.
        """
        assert upgrade_kueue_stopped_notebook_statefulset.exists, (
            f"StatefulSet '{upgrade_kueue_stopped_notebook_statefulset.name}' no longer exists after upgrade"
        )
        replicas = upgrade_kueue_stopped_notebook_statefulset.instance.spec.replicas
        assert replicas == 0, f"Kueue stopped notebook StatefulSet has {replicas} replicas after upgrade, expected 0"

    @pytest.mark.post_upgrade
    def test_kueue_stopped_notebook_pod_absent_after_upgrade(
        self,
        upgrade_kueue_stopped_notebook: Notebook,
        unprivileged_client: DynamicClient,
    ) -> None:
        """Given a kueue notebook was stopped before upgrade,
        When the upgrade completes,
        Then no pod should exist for the stopped notebook.
        """
        notebook_pod = Pod(
            client=unprivileged_client,
            namespace=upgrade_kueue_stopped_notebook.namespace,
            name=f"{upgrade_kueue_stopped_notebook.name}-0",
        )
        assert not notebook_pod.exists, (
            f"Pod '{notebook_pod.name}' unexpectedly exists after upgrade "
            f"for a kueue notebook that was stopped before upgrade"
        )


@pytest.mark.usefixtures("kueue_statefulset_framework_check")
class TestPostUpgradeKueueCreation:
    """Verify a new kueue-managed notebook can be created on the upgraded platform.

    Steps:
        1. Create a fresh Notebook CR with kueue queue-name label post-upgrade.
        2. Verify the pod reaches Ready state.
        3. Verify Kueue applies management labels to the new pod.
    """

    @pytest.mark.post_upgrade
    def test_new_kueue_notebook_pod_ready(
        self,
        new_kueue_notebook_pod: Pod,
    ) -> None:
        """Given the platform was upgraded,
        When a new kueue-managed Notebook CR is created,
        Then the notebook pod should reach Ready state.

        Validation is performed by the new_kueue_notebook_pod fixture.
        """

    @pytest.mark.post_upgrade
    def test_new_kueue_notebook_has_management_labels(
        self,
        new_kueue_notebook_pod: Pod,
    ) -> None:
        """Given a new kueue-managed notebook is created post-upgrade,
        When Kueue admits the workload,
        Then the pod should have kueue.x-k8s.io/managed=true label.
        """
        pod_labels = new_kueue_notebook_pod.instance.metadata.labels or {}
        assert pod_labels.get(KUEUE_MANAGED_LABEL) == "true", (
            f"New kueue notebook pod should have '{KUEUE_MANAGED_LABEL}=true' label "
            f"on upgraded platform. Labels: {list(pod_labels.keys())}"
        )

    @pytest.mark.post_upgrade
    def test_new_kueue_notebook_has_queue_name_label(
        self,
        new_kueue_notebook_pod: Pod,
    ) -> None:
        """Given a new kueue-managed notebook is created post-upgrade,
        When Kueue admits the workload,
        Then the pod should reference the correct LocalQueue.
        """
        pod_labels = new_kueue_notebook_pod.instance.metadata.labels or {}
        assert pod_labels.get(KUEUE_QUEUE_NAME_LABEL) == UPGRADE_KUEUE_LOCAL_QUEUE_NAME, (
            f"New kueue notebook pod should have "
            f"'{KUEUE_QUEUE_NAME_LABEL}={UPGRADE_KUEUE_LOCAL_QUEUE_NAME}' label. "
            f"Got: {pod_labels.get(KUEUE_QUEUE_NAME_LABEL)}"
        )

    @pytest.mark.post_upgrade
    def test_new_kueue_notebook_has_cluster_queue_label(
        self,
        new_kueue_notebook_pod: Pod,
    ) -> None:
        """Given a new kueue-managed notebook is created post-upgrade,
        When Kueue admits the workload,
        Then the pod should have the cluster-queue-name label.
        """
        pod_labels = new_kueue_notebook_pod.instance.metadata.labels or {}
        assert pod_labels.get(KUEUE_CLUSTER_QUEUE_LABEL) == UPGRADE_KUEUE_CLUSTER_QUEUE_NAME, (
            f"New kueue notebook pod should have "
            f"'{KUEUE_CLUSTER_QUEUE_LABEL}={UPGRADE_KUEUE_CLUSTER_QUEUE_NAME}' label. "
            f"Got: {pod_labels.get(KUEUE_CLUSTER_QUEUE_LABEL)}"
        )

    @pytest.mark.post_upgrade
    def test_new_kueue_notebook_has_local_queue_label(
        self,
        new_kueue_notebook_pod: Pod,
    ) -> None:
        """Given a new kueue-managed notebook is created post-upgrade,
        When Kueue admits the workload,
        Then the pod should have the local-queue-name label.
        """
        pod_labels = new_kueue_notebook_pod.instance.metadata.labels or {}
        assert pod_labels.get(KUEUE_LOCAL_QUEUE_LABEL) == UPGRADE_KUEUE_LOCAL_QUEUE_NAME, (
            f"New kueue notebook pod should have "
            f"'{KUEUE_LOCAL_QUEUE_LABEL}={UPGRADE_KUEUE_LOCAL_QUEUE_NAME}' label. "
            f"Got: {pod_labels.get(KUEUE_LOCAL_QUEUE_LABEL)}"
        )

    @pytest.mark.post_upgrade
    def test_new_kueue_notebook_workload_admitted(
        self,
        admin_client: DynamicClient,
        new_kueue_notebook: Notebook,
        upgrade_kueue_namespace: Namespace,
    ) -> None:
        """Given a new kueue-managed notebook is created post-upgrade,
        When Kueue processes the workload,
        Then a Workload object should exist with Admitted=True condition
            and correct ClusterQueue assignment.
        """
        workloads = list(Workload.get(client=admin_client, namespace=upgrade_kueue_namespace.name))
        notebook_workloads = [wl for wl in workloads if new_kueue_notebook.name in wl.name]
        assert notebook_workloads, (
            f"Kueue should create a Workload object for new notebook '{new_kueue_notebook.name}' post-upgrade"
        )

        workload = notebook_workloads[0]
        conditions = workload.instance.status.get("conditions", [])
        is_admitted = any(c["type"] == "Admitted" and c["status"] == "True" for c in conditions)
        assert is_admitted, f"Workload '{workload.name}' for new notebook should be Admitted. Conditions: {conditions}"

        admission = workload.instance.status.get("admission", {})
        assert admission.get("clusterQueue") == UPGRADE_KUEUE_CLUSTER_QUEUE_NAME, (
            f"Workload should be admitted to ClusterQueue "
            f"'{UPGRADE_KUEUE_CLUSTER_QUEUE_NAME}', "
            f"got: '{admission.get('clusterQueue')}'"
        )
