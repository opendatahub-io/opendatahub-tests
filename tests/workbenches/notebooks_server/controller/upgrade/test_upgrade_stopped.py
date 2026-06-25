from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.notebook import Notebook
from ocp_resources.pod import Pod

from tests.workbenches.notebooks_server.controller.utils import StatefulSet


@pytest.mark.usefixtures("stopped_notebook_pre_upgrade_shutdown")
class TestPreUpgradeStoppedNotebook:
    """Verify a stopped notebook remains scaled down before the platform upgrade.

    Steps:
        1. Create a Notebook CR, wait for Ready, then stop it via annotation.
        2. Verify the StatefulSet has replicas=0.
        3. Verify no pod exists for the stopped notebook.
    """

    @pytest.mark.pre_upgrade
    def test_stopped_notebook_has_zero_replicas(
        self,
        stopped_notebook_statefulset: StatefulSet,
    ) -> None:
        """Given a notebook was stopped via kubeflow-resource-stopped annotation,
        When the controller reconciles the StatefulSet,
        Then the StatefulSet should have 0 replicas.
        """
        replicas = stopped_notebook_statefulset.instance.spec.replicas
        assert replicas == 0, f"StatefulSet '{stopped_notebook_statefulset.name}' has {replicas} replicas, expected 0"

    @pytest.mark.pre_upgrade
    def test_stopped_notebook_pod_absent(
        self,
        stopped_notebook: Notebook,
        unprivileged_client: DynamicClient,
    ) -> None:
        """Given a notebook was stopped via kubeflow-resource-stopped annotation,
        When the pod terminates,
        Then no pod should exist for the stopped notebook.
        """
        notebook_pod = Pod(
            client=unprivileged_client,
            namespace=stopped_notebook.namespace,
            name=f"{stopped_notebook.name}-0",
        )
        assert not notebook_pod.exists, f"Pod '{notebook_pod.name}' still exists for stopped notebook"


class TestPostUpgradeStoppedNotebook:
    """Verify a stopped notebook remains stopped after the platform upgrade.

    Steps:
        1. Verify the stop annotation is still present.
        2. Verify the stop annotation value (timestamp) was not modified.
        3. Verify the StatefulSet still has replicas=0.
        4. Verify no pod was (re)created for the stopped notebook.
    """

    @pytest.mark.post_upgrade
    def test_stopped_annotation_preserved_after_upgrade(
        self,
        stopped_notebook: Notebook,
    ) -> None:
        """Given a notebook was stopped before upgrade,
        When the upgrade completes,
        Then the kubeflow-resource-stopped annotation should still be present.
        """
        stop_annotation = stopped_notebook.instance.metadata.annotations.get("kubeflow-resource-stopped")
        assert stop_annotation is not None, (
            f"Notebook '{stopped_notebook.name}' lost 'kubeflow-resource-stopped' annotation after upgrade. "
            f"Current annotations: {list(stopped_notebook.instance.metadata.annotations.keys())}"
        )

    @pytest.mark.post_upgrade
    def test_stopped_annotation_value_unchanged_after_upgrade(
        self,
        stopped_notebook: Notebook,
        upgrade_notebook_baseline: dict[str, Any],
    ) -> None:
        """Given a notebook was stopped with a specific timestamp before upgrade,
        When the upgrade completes,
        Then the annotation value should be unchanged.
        """
        current_value = stopped_notebook.instance.metadata.annotations.get("kubeflow-resource-stopped")
        saved_value = upgrade_notebook_baseline["stopped_annotation_value"]

        assert current_value == saved_value, (
            f"Annotation 'kubeflow-resource-stopped' value changed during upgrade. "
            f"Pre-upgrade: '{saved_value}', post-upgrade: '{current_value}'"
        )

    @pytest.mark.post_upgrade
    def test_stopped_notebook_still_has_zero_replicas(
        self,
        stopped_notebook_statefulset: StatefulSet,
    ) -> None:
        """Given a notebook was stopped before upgrade,
        When the upgrade completes,
        Then the StatefulSet should still have 0 replicas.
        """
        assert stopped_notebook_statefulset.exists, (
            f"StatefulSet '{stopped_notebook_statefulset.name}' no longer exists after upgrade"
        )
        replicas = stopped_notebook_statefulset.instance.spec.replicas
        assert replicas == 0, (
            f"StatefulSet '{stopped_notebook_statefulset.name}' has {replicas} replicas after upgrade, "
            f"expected 0 (notebook was stopped before upgrade)"
        )

    @pytest.mark.post_upgrade
    def test_stopped_notebook_pod_absent_after_upgrade(
        self,
        stopped_notebook: Notebook,
        unprivileged_client: DynamicClient,
    ) -> None:
        """Given a notebook was stopped before upgrade,
        When the upgrade completes,
        Then no pod should exist for the stopped notebook.
        """
        notebook_pod = Pod(
            client=unprivileged_client,
            namespace=stopped_notebook.namespace,
            name=f"{stopped_notebook.name}-0",
        )
        assert not notebook_pod.exists, (
            f"Pod '{notebook_pod.name}' unexpectedly exists after upgrade for a notebook "
            f"that was stopped before upgrade"
        )
