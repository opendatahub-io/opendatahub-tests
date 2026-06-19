import json
from typing import Any

import pytest
from ocp_resources.config_map import ConfigMap
from ocp_resources.notebook import Notebook
from ocp_resources.pod import Pod
from ocp_resources.service import Service

from tests.workbenches.notebooks_server.controller.utils import CA_BUNDLE_CERT_KEY, StatefulSet


@pytest.mark.usefixtures("capture_notebook_baseline")
class TestPreUpgradeNotebook:
    """Verify a workbench notebook is running before the platform upgrade.

    Steps:
        1. Create a Notebook CR with PVC and supporting resources.
        2. Wait for the notebook pod to reach Ready state.
        3. Capture resource metadata to a ConfigMap for post-upgrade comparison.
    """

    @pytest.mark.pre_upgrade
    def test_notebook_running_before_upgrade(self, upgrade_notebook_pod: Pod) -> None:
        """Given a Notebook CR is created before upgrade,
        When the notebook controller reconciles and starts the pod,
        Then the notebook pod should exist and be in Ready state.

        Validation is performed by the upgrade_notebook_pod fixture which waits
        for the pod to exist and reach Ready condition.
        """

    @pytest.mark.pre_upgrade
    def test_ca_bundle_configmap_exists_before_upgrade(
        self,
        workbench_trusted_ca_bundle: ConfigMap,
    ) -> None:
        """Given a notebook is running before upgrade,
        When the ODH controller reconciles CA certificates,
        Then the workbench-trusted-ca-bundle ConfigMap should exist with ca-bundle.crt.
        """
        assert workbench_trusted_ca_bundle.exists, (
            f"ConfigMap '{workbench_trusted_ca_bundle.name}' not found in namespace "
            f"'{workbench_trusted_ca_bundle.namespace}'"
        )

        cm_data = workbench_trusted_ca_bundle.instance.data
        assert cm_data and cm_data.get(CA_BUNDLE_CERT_KEY) is not None, (
            f"ConfigMap '{workbench_trusted_ca_bundle.name}' missing key '{CA_BUNDLE_CERT_KEY}'."
        )


class TestPostUpgradeNotebook:
    """Verify the workbench notebook survived the platform upgrade.

    Steps:
        1. Verify the notebook pod still exists after upgrade.
        2. Compare the pod's creationTimestamp against the pre-upgrade baseline.
        3. Verify Notebook CR, StatefulSet, and Service were not modified.
    """

    @pytest.mark.post_upgrade
    def test_notebook_not_restarted_after_upgrade(
        self,
        upgrade_notebook_pod: Pod,
        upgrade_notebook_baseline: dict[str, Any],
    ) -> None:
        """Given a notebook was running before upgrade,
        When the upgrade completes,
        Then the pod's creationTimestamp should match the pre-upgrade baseline.
        """
        assert upgrade_notebook_pod.exists, f"Notebook pod '{upgrade_notebook_pod.name}' no longer exists after upgrade"

        current_timestamp = upgrade_notebook_pod.instance.metadata.creationTimestamp
        saved_timestamp = upgrade_notebook_baseline["ntb_creation_timestamp"]

        assert current_timestamp == saved_timestamp, (
            f"Notebook pod was restarted during upgrade. "
            f"Pre-upgrade creationTimestamp: {saved_timestamp}, "
            f"post-upgrade creationTimestamp: {current_timestamp}"
        )

    @pytest.mark.post_upgrade
    def test_notebook_cr_not_modified_after_upgrade(
        self,
        upgrade_notebook: Notebook,
        upgrade_notebook_baseline: dict[str, Any],
    ) -> None:
        """Given a Notebook CR existed before upgrade,
        When the upgrade completes,
        Then the Notebook CR generation should be unchanged.
        """
        current_generation = upgrade_notebook.instance.metadata.generation
        saved_generation = upgrade_notebook_baseline["notebook_generation"]

        assert current_generation == saved_generation, (
            f"Notebook CR was modified during upgrade. "
            f"Pre-upgrade generation: {saved_generation}, "
            f"post-upgrade generation: {current_generation}"
        )

    @pytest.mark.post_upgrade
    def test_statefulset_not_modified_after_upgrade(
        self,
        upgrade_notebook_statefulset: StatefulSet,
        upgrade_notebook_baseline: dict[str, Any],
    ) -> None:
        """Given a notebook StatefulSet existed before upgrade,
        When the upgrade completes,
        Then the StatefulSet generation should be unchanged.
        """
        assert upgrade_notebook_statefulset.exists, (
            f"StatefulSet '{upgrade_notebook_statefulset.name}' no longer exists after upgrade"
        )

        current_generation = upgrade_notebook_statefulset.instance.metadata.generation
        saved_generation = upgrade_notebook_baseline["statefulset_generation"]

        assert current_generation == saved_generation, (
            f"StatefulSet was modified during upgrade. "
            f"Pre-upgrade generation: {saved_generation}, "
            f"post-upgrade generation: {current_generation}"
        )

    @pytest.mark.post_upgrade
    def test_statefulset_healthy_after_upgrade(
        self,
        upgrade_notebook_statefulset: StatefulSet,
    ) -> None:
        """Given a notebook StatefulSet existed before upgrade,
        When the upgrade completes,
        Then readyReplicas should equal spec.replicas and no rollout should be pending.
        """
        assert upgrade_notebook_statefulset.exists, (
            f"StatefulSet '{upgrade_notebook_statefulset.name}' no longer exists after upgrade"
        )
        sts = upgrade_notebook_statefulset.instance
        expected_replicas = sts.spec.replicas
        ready_replicas = sts.status.readyReplicas or 0

        assert ready_replicas == expected_replicas, (
            f"StatefulSet '{upgrade_notebook_statefulset.name}' has {ready_replicas} ready replicas, "
            f"expected {expected_replicas}. The notebook pod may be in a degraded state "
            f"that the StatefulSet cannot recover from without manual intervention."
        )

        current_revision = sts.status.currentRevision
        update_revision = sts.status.updateRevision
        assert current_revision == update_revision, (
            f"StatefulSet '{upgrade_notebook_statefulset.name}' has a pending rollout: "
            f"currentRevision='{current_revision}', updateRevision='{update_revision}'"
        )

    @pytest.mark.post_upgrade
    def test_service_not_modified_after_upgrade(
        self,
        upgrade_notebook_service: Service,
        upgrade_notebook_baseline: dict[str, Any],
    ) -> None:
        """Given a notebook Service existed before upgrade,
        When the upgrade completes,
        Then the Service spec (ports, selector) should be unchanged.
        """
        assert upgrade_notebook_service.exists, (
            f"Service '{upgrade_notebook_service.name}' no longer exists after upgrade"
        )

        service_spec = upgrade_notebook_service.instance.spec
        current_ports = json.dumps(service_spec.ports, sort_keys=True, default=str)
        current_selector = json.dumps(service_spec.selector, sort_keys=True, default=str)

        saved_ports = upgrade_notebook_baseline["service_ports"]
        saved_selector = upgrade_notebook_baseline["service_selector"]

        assert current_ports == saved_ports, (
            f"Service ports were modified during upgrade. Pre-upgrade: {saved_ports}, post-upgrade: {current_ports}"
        )

        assert current_selector == saved_selector, (
            f"Service selector was modified during upgrade. "
            f"Pre-upgrade: {saved_selector}, "
            f"post-upgrade: {current_selector}"
        )

    @pytest.mark.post_upgrade
    def test_ca_bundle_configmap_exists_after_upgrade(
        self,
        workbench_trusted_ca_bundle: ConfigMap,
    ) -> None:
        """Given the workbench-trusted-ca-bundle existed before upgrade,
        When the upgrade completes,
        Then the ConfigMap should still exist with the ca-bundle.crt key.
        """
        assert workbench_trusted_ca_bundle.exists, (
            f"ConfigMap '{workbench_trusted_ca_bundle.name}' no longer exists after upgrade"
        )

        cm_data = workbench_trusted_ca_bundle.instance.data
        assert cm_data and cm_data.get(CA_BUNDLE_CERT_KEY) is not None, (
            f"ConfigMap '{workbench_trusted_ca_bundle.name}' lost '{CA_BUNDLE_CERT_KEY}' key after upgrade."
        )

    @pytest.mark.post_upgrade
    def test_ca_bundle_configmap_consistency_after_upgrade(
        self,
        workbench_trusted_ca_bundle: ConfigMap,
        odh_trusted_ca_bundle: ConfigMap,
        upgrade_notebook_baseline: dict[str, Any],
    ) -> None:
        """Given the workbench-trusted-ca-bundle and odh-trusted-ca-bundle existed before upgrade,
        When the upgrade completes,
        Then both CA bundles must change together or neither should change.
        """
        assert workbench_trusted_ca_bundle.exists, (
            f"ConfigMap '{workbench_trusted_ca_bundle.name}' no longer exists after upgrade"
        )
        assert odh_trusted_ca_bundle.exists, f"ConfigMap '{odh_trusted_ca_bundle.name}' no longer exists after upgrade"

        saved_workbench_rv = upgrade_notebook_baseline["ca_bundle_resource_version"]
        saved_odh_rv = upgrade_notebook_baseline["odh_ca_bundle_resource_version"]

        current_workbench_rv = workbench_trusted_ca_bundle.instance.metadata.resourceVersion
        current_odh_rv = odh_trusted_ca_bundle.instance.metadata.resourceVersion

        odh_bundle_changed = current_odh_rv != saved_odh_rv
        workbench_bundle_changed = current_workbench_rv != saved_workbench_rv

        if odh_bundle_changed:
            assert workbench_bundle_changed, (
                f"odh-trusted-ca-bundle was modified during upgrade but the change was not "
                f"propagated to workbench-trusted-ca-bundle. "
                f"odh resourceVersion: {saved_odh_rv} -> {current_odh_rv}, "
                f"workbench resourceVersion: {saved_workbench_rv} (unchanged)"
            )
        else:
            assert not workbench_bundle_changed, (
                f"workbench-trusted-ca-bundle was modified during upgrade but the source "
                f"odh-trusted-ca-bundle was not. This indicates an unexpected reconciliation. "
                f"workbench resourceVersion: {saved_workbench_rv} -> {current_workbench_rv}, "
                f"odh resourceVersion: {saved_odh_rv} (unchanged)"
            )
