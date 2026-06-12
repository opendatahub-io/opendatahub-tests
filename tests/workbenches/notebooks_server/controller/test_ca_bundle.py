import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.namespace import Namespace
from ocp_resources.notebook import Notebook
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.workbenches.notebooks_server.controller.utils import (
    CA_BUNDLE_CERT_KEY,
    WORKBENCH_TRUSTED_CA_BUNDLE_NAME,
)
from utilities.constants import Timeout

NOTEBOOK_NAME = "test-ca-bundle"
NOTEBOOK_PVC_SIZE = "1Gi"
# Enough for an idle Jupyter + kube-rbac-proxy sidecar; this test only checks pod identity.
NOTEBOOK_RESOURCES: dict[str, dict[str, str]] = {
    "limits": {"cpu": "500m", "memory": "1Gi"},
    "requests": {"cpu": "100m", "memory": "512Mi"},
}
# Observe that the pod UID stays stable for this window after the CA ConfigMap patch.
UID_STABILITY_OBSERVE_SECONDS = 30
UID_STABILITY_POLL_SECONDS = 5


class TestCaBundleResilience:
    """Verify that CA bundle changes do not cause unnecessary workbench pod restarts."""

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        "unprivileged_model_namespace,users_persistent_volume_claim,default_notebook,notebook_pod",
        [
            pytest.param(
                {
                    "name": NOTEBOOK_NAME,
                    "add-dashboard-label": True,
                },
                {"name": NOTEBOOK_NAME, "size": NOTEBOOK_PVC_SIZE},
                {
                    "namespace": NOTEBOOK_NAME,
                    "name": NOTEBOOK_NAME,
                    "resources": NOTEBOOK_RESOURCES,
                },
                {"timeout": Timeout.TIMEOUT_2MIN},
                id="test_ca_bundle_update_does_not_restart_pod",
            )
        ],
        indirect=True,
    )
    def test_ca_bundle_update_does_not_restart_pod(
        self,
        admin_client: DynamicClient,
        notebook_pod: Pod,
        unprivileged_model_namespace: Namespace,
        users_persistent_volume_claim: PersistentVolumeClaim,
        default_notebook: Notebook,
    ) -> None:
        """Given a notebook is running with a workbench-trusted-ca-bundle ConfigMap,
        When the CA bundle ConfigMap is modified,
        Then the notebook pod should NOT be restarted.
        """
        ca_bundle_cm = ConfigMap(
            client=admin_client,
            name=WORKBENCH_TRUSTED_CA_BUNDLE_NAME,
            namespace=unprivileged_model_namespace.name,
        )
        try:
            for cm_ready in TimeoutSampler(
                wait_timeout=Timeout.TIMEOUT_1MIN,
                sleep=UID_STABILITY_POLL_SECONDS,
                func=lambda: bool(ca_bundle_cm.exists and (ca_bundle_cm.instance.data or {}).get(CA_BUNDLE_CERT_KEY)),
            ):
                if cm_ready:
                    break
        except TimeoutExpiredError as exp:
            raise AssertionError(
                f"ConfigMap '{WORKBENCH_TRUSTED_CA_BUNDLE_NAME}' with key '{CA_BUNDLE_CERT_KEY}' "
                f"not found in '{unprivileged_model_namespace.name}' within {Timeout.TIMEOUT_1MIN}s"
            ) from exp

        pod_uid_before = notebook_pod.instance.metadata.uid
        assert pod_uid_before, f"Pod '{notebook_pod.name}' has no UID"

        cm_data = ca_bundle_cm.instance.data or {}
        original_cert = cm_data.get(CA_BUNDLE_CERT_KEY, "")
        patched_cert = original_cert + "\n# test-ca-bundle-update-marker"

        ResourceEditor(patches={ca_bundle_cm: {"data": {CA_BUNDLE_CERT_KEY: patched_cert}}}).update()

        notebook_pod.wait_for_condition(
            condition=Pod.Condition.READY,
            status=Pod.Condition.Status.TRUE,
            timeout=Timeout.TIMEOUT_1MIN,
        )

        restart_message = (
            "Notebook pod was restarted after CA bundle ConfigMap update. "
            f"Pod UID before: {pod_uid_before}, after: {{pod_uid_after}}. "
            f"Updating '{WORKBENCH_TRUSTED_CA_BUNDLE_NAME}' must not restart the workbench pod."
        )
        try:
            for current_uid in TimeoutSampler(
                wait_timeout=UID_STABILITY_OBSERVE_SECONDS,
                sleep=UID_STABILITY_POLL_SECONDS,
                func=lambda: notebook_pod.instance.metadata.uid,
            ):
                assert current_uid == pod_uid_before, restart_message.format(pod_uid_after=current_uid)
        except TimeoutExpiredError:
            # Observation window completed without a UID change — expected success path.
            pass
