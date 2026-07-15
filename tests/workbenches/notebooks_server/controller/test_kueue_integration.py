"""Integration tests for Kueue scheduling of Notebook workbenches via HardwareProfiles.

Verifies that Notebook CRs annotated with a HardwareProfile that has
scheduling.type=Queue are properly admitted, scheduled, and resource-tracked
by the Red Hat build of Kueue operator.

The HardwareProfile webhook injects:
- kueue.x-k8s.io/queue-name label (from scheduling.kueue.localQueueName)
- Container resources (from identifiers[].defaultCount)

Prerequisites:
    - Red Hat build of Kueue operator installed with 'StatefulSet' in frameworks
    - DSC Kueue component set to Unmanaged
    - Namespaces labeled with kueue.openshift.io/managed=true
    - HardwareProfile CRD and ODH webhook active
"""

from datetime import UTC, datetime
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.notebook import Notebook
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.workbenches.notebooks_server.controller.utils import HardwareProfile
from utilities.constants import Timeout
from utilities.kueue_utils import (
    ClusterQueue,
    LocalQueue,
    ResourceFlavor,
    Workload,
    check_gated_pods_and_running_pods,
)

LOGGER = structlog.get_logger(name=__name__)

_KUEUE_QUEUE_NAME_LABEL = "kueue.x-k8s.io/queue-name"
_KUEUE_MANAGED_LABEL = "kueue.x-k8s.io/managed"
_KUEUE_CLUSTER_QUEUE_LABEL = "kueue.x-k8s.io/cluster-queue-name"
_KUEUE_LOCAL_QUEUE_LABEL = "kueue.x-k8s.io/local-queue-name"
_KUBEFLOW_STOPPED_ANNOTATION = "kubeflow-resource-stopped"

pytestmark = [
    pytest.mark.kueue,
    pytest.mark.smoke,
]


def _get_notebook_workload(client: DynamicClient, namespace: str, notebook_name: str) -> Workload | None:
    """Find the Kueue Workload object associated with a notebook."""
    workloads = list(Workload.get(client=client, namespace=namespace))
    matching = [wl for wl in workloads if notebook_name in wl.name]
    return matching[0] if matching else None


def _workload_is_admitted(workload: Workload) -> bool:
    """Check whether a Workload has the 'Admitted: True' condition."""
    conditions = workload.instance.status.get("conditions", [])
    return any(c["type"] == "Admitted" and c["status"] == "True" for c in conditions)


def _assert_kueue_pod_labels(pod: Pod, cluster_queue_name: str, local_queue_name: str) -> None:
    """Assert that a pod carries the full set of Kueue scheduling labels."""
    labels = pod.instance.metadata.labels or {}
    assert labels.get(_KUEUE_MANAGED_LABEL) == "true", (
        f"Pod should have '{_KUEUE_MANAGED_LABEL}=true'. Labels: {list(labels.keys())}"
    )
    assert labels.get(_KUEUE_CLUSTER_QUEUE_LABEL) == cluster_queue_name, (
        f"Pod should have cluster-queue label '{cluster_queue_name}', got: '{labels.get(_KUEUE_CLUSTER_QUEUE_LABEL)}'"
    )
    assert labels.get(_KUEUE_LOCAL_QUEUE_LABEL) == local_queue_name, (
        f"Pod should have local-queue label '{local_queue_name}', got: '{labels.get(_KUEUE_LOCAL_QUEUE_LABEL)}'"
    )


class _NormalScenario:
    """Configuration constants for the normal-resources scenario."""

    NAMESPACE_NAME = "test-kueue-notebook"
    LOCAL_QUEUE_NAME = "notebook-local-queue"
    CLUSTER_QUEUE_NAME = "notebook-cluster-queue"
    RESOURCE_FLAVOR_NAME = "notebook-flavor"
    CPU_QUOTA = "4"
    MEMORY_QUOTA = "8Gi"
    NOTEBOOK_NAME = "test-kueue-notebook"
    HWP_NAME = "notebook-hwp"
    HWP_CPU_DEFAULT = "500m"
    HWP_MEMORY_DEFAULT = "1Gi"


class _StarvationScenario:
    """Configuration constants for the resource-starvation scenario."""

    NAMESPACE_NAME = "test-kueue-low-resource"
    LOCAL_QUEUE_NAME = "low-resource-local-queue"
    CLUSTER_QUEUE_NAME = "low-resource-cluster-queue"
    RESOURCE_FLAVOR_NAME = "low-resource-flavor"
    CPU_QUOTA = "100m"
    MEMORY_QUOTA = "64Mi"
    NOTEBOOK_NAME = "high-demand-notebook"
    HWP_NAME = "high-demand-hwp"
    HWP_CPU_DEFAULT = "2"
    HWP_MEMORY_DEFAULT = "2Gi"


def _normal_resources_params() -> tuple[Any, ...]:
    """Fixture params for the normal-resources test scenario."""
    return (
        {"name": _NormalScenario.NAMESPACE_NAME},
        {"name": _NormalScenario.NOTEBOOK_NAME},
        {"name": _NormalScenario.RESOURCE_FLAVOR_NAME},
        {
            "name": _NormalScenario.CLUSTER_QUEUE_NAME,
            "cpu_quota": _NormalScenario.CPU_QUOTA,
            "memory_quota": _NormalScenario.MEMORY_QUOTA,
            "namespace_selector": {"matchLabels": {"kubernetes.io/metadata.name": _NormalScenario.NAMESPACE_NAME}},
        },
        {"name": _NormalScenario.LOCAL_QUEUE_NAME},
        {
            "name": _NormalScenario.HWP_NAME,
            "cpu_default": _NormalScenario.HWP_CPU_DEFAULT,
            "memory_default": _NormalScenario.HWP_MEMORY_DEFAULT,
        },
        {"name": _NormalScenario.NOTEBOOK_NAME},
    )


def _resource_starvation_params() -> tuple[Any, ...]:
    """Fixture params for the resource-starvation scenario (quota too low for notebook)."""
    return (
        {"name": _StarvationScenario.NAMESPACE_NAME},
        {"name": _StarvationScenario.NOTEBOOK_NAME},
        {"name": _StarvationScenario.RESOURCE_FLAVOR_NAME},
        {
            "name": _StarvationScenario.CLUSTER_QUEUE_NAME,
            "cpu_quota": _StarvationScenario.CPU_QUOTA,
            "memory_quota": _StarvationScenario.MEMORY_QUOTA,
            "namespace_selector": {"matchLabels": {"kubernetes.io/metadata.name": _StarvationScenario.NAMESPACE_NAME}},
        },
        {"name": _StarvationScenario.LOCAL_QUEUE_NAME},
        {
            "name": _StarvationScenario.HWP_NAME,
            "cpu_default": _StarvationScenario.HWP_CPU_DEFAULT,
            "memory_default": _StarvationScenario.HWP_MEMORY_DEFAULT,
        },
        {"name": _StarvationScenario.NOTEBOOK_NAME},
    )


_FIXTURE_NAMES = (
    "kueue_notebook_namespace,"
    "kueue_notebook_pvc,"
    "kueue_resource_flavor,"
    "kueue_cluster_queue,"
    "kueue_local_queue,"
    "kueue_hardware_profile,"
    "kueue_notebook"
)


@pytest.mark.usefixtures("kueue_statefulset_framework_check")
class TestKueueNotebookIntegration:
    """Test Kueue integration with Notebook workbenches.

    Verifies admission control, resource tracking, and lifecycle management
    for notebooks scheduled through Kueue queues.
    """

    @pytest.mark.parametrize(
        _FIXTURE_NAMES,
        [pytest.param(*_normal_resources_params(), id="test_admission_control")],
        indirect=True,
    )
    def test_kueue_notebook_admission_control(
        self,
        admin_client: DynamicClient,
        unprivileged_client: DynamicClient,
        kueue_notebook_namespace: Namespace,
        kueue_notebook_pvc: PersistentVolumeClaim,
        kueue_resource_flavor: ResourceFlavor,
        kueue_cluster_queue: ClusterQueue,
        kueue_local_queue: LocalQueue,
        kueue_hardware_profile: HardwareProfile,
        kueue_notebook: Notebook,
    ) -> None:
        """Verify Kueue admits a notebook workload via HardwareProfile scheduling.

        Given a HardwareProfile with scheduling.type=Queue referencing a LocalQueue,
        When a Notebook CR is created with the HWP annotation (no explicit resources/labels),
        Then the webhook injects the queue-name label and resources from the HWP,
            Kueue creates an admitted Workload, reserves quota in the correct
            ClusterQueue, and labels the pod with full scheduling metadata.
        """
        assert kueue_notebook.exists, "Notebook CR should be created successfully"

        notebook_labels = kueue_notebook.instance.metadata.labels or {}
        assert notebook_labels.get(_KUEUE_QUEUE_NAME_LABEL) == kueue_local_queue.name, (
            f"Webhook should inject queue-name label '{kueue_local_queue.name}' from HWP scheduling config, "
            f"got: {notebook_labels.get(_KUEUE_QUEUE_NAME_LABEL)}"
        )

        notebook_container = kueue_notebook.instance.spec.template.spec.containers[0]
        injected_resources = notebook_container.resources
        assert injected_resources, "Webhook should inject container resources from HWP identifiers"
        assert injected_resources.requests.get("cpu") == _NormalScenario.HWP_CPU_DEFAULT, (
            f"Webhook should inject CPU request '{_NormalScenario.HWP_CPU_DEFAULT}' from HWP defaultCount, "
            f"got: '{injected_resources.requests.get('cpu')}'"
        )
        assert injected_resources.requests.get("memory") == _NormalScenario.HWP_MEMORY_DEFAULT, (
            f"Webhook should inject memory request '{_NormalScenario.HWP_MEMORY_DEFAULT}' from HWP defaultCount, "
            f"got: '{injected_resources.requests.get('memory')}'"
        )

        notebook_pod = Pod(
            client=unprivileged_client,
            namespace=kueue_notebook_namespace.name,
            name=f"{kueue_notebook.name}-0",
        )
        notebook_pod.wait(timeout=Timeout.TIMEOUT_2MIN)
        assert notebook_pod.exists, f"Notebook pod '{notebook_pod.name}' should be created"

        _assert_kueue_pod_labels(
            pod=notebook_pod,
            cluster_queue_name=kueue_cluster_queue.name,
            local_queue_name=kueue_local_queue.name,
        )

        notebook_pod.wait_for_condition(
            condition=Pod.Condition.READY,
            status=Pod.Condition.Status.TRUE,
            timeout=Timeout.TIMEOUT_5MIN,
        )
        LOGGER.info(f"Notebook pod '{notebook_pod.name}' admitted and running under Kueue management via HWP")

        workload = _get_notebook_workload(
            client=admin_client,
            namespace=kueue_notebook_namespace.name,
            notebook_name=kueue_notebook.name,
        )
        assert workload, f"Kueue should create a Workload object for notebook '{kueue_notebook.name}'"
        assert _workload_is_admitted(workload), (
            f"Workload '{workload.name}' should have 'Admitted: True' condition. "
            f"Conditions: {workload.instance.status.get('conditions', [])}"
        )

        admission = workload.instance.status.get("admission", {})
        assert admission.get("clusterQueue") == kueue_cluster_queue.name, (
            f"Workload should be admitted to ClusterQueue '{kueue_cluster_queue.name}', "
            f"got: '{admission.get('clusterQueue')}'"
        )

    @pytest.mark.parametrize(
        _FIXTURE_NAMES,
        [pytest.param(*_resource_starvation_params(), id="test_resource_starvation")],
        indirect=True,
    )
    def test_kueue_notebook_resource_constraints(
        self,
        admin_client: DynamicClient,
        unprivileged_client: DynamicClient,
        kueue_notebook_namespace: Namespace,
        kueue_notebook_pvc: PersistentVolumeClaim,
        kueue_resource_flavor: ResourceFlavor,
        kueue_cluster_queue: ClusterQueue,
        kueue_local_queue: LocalQueue,
        kueue_hardware_profile: HardwareProfile,
        kueue_notebook: Notebook,
    ) -> None:
        """Verify Kueue gates a notebook pod when HWP resources exceed quota.

        Given a ClusterQueue with very low CPU/memory quotas and a HardwareProfile
            whose defaultCount values exceed the quota,
        When a Notebook CR is created with the HWP annotation,
        Then the webhook injects resources exceeding quota, the pod remains
            in SchedulingGated state, and the Workload is NOT admitted.
        """
        assert kueue_notebook.exists, "Notebook CR should be created successfully"

        notebook_pod = Pod(
            client=unprivileged_client,
            namespace=kueue_notebook_namespace.name,
            name=f"{kueue_notebook.name}-0",
        )
        notebook_pod.wait(timeout=Timeout.TIMEOUT_2MIN)
        assert notebook_pod.exists, f"Notebook pod '{notebook_pod.name}' should exist"

        try:
            for sample in TimeoutSampler(
                wait_timeout=30,
                sleep=2,
                func=check_gated_pods_and_running_pods,
                labels=[f"{_KUEUE_QUEUE_NAME_LABEL}={kueue_local_queue.name}"],
                namespace=kueue_notebook_namespace.name,
                admin_client=admin_client,
            ):
                running_pods, gated_pods = sample
                if gated_pods >= 1:
                    break
        except TimeoutExpiredError:
            running_pods, gated_pods = check_gated_pods_and_running_pods(
                labels=[f"{_KUEUE_QUEUE_NAME_LABEL}={kueue_local_queue.name}"],
                namespace=kueue_notebook_namespace.name,
                admin_client=admin_client,
            )

        assert gated_pods >= 1, (
            f"Notebook pod should be gated due to insufficient quota "
            f"(CPU quota: {_StarvationScenario.CPU_QUOTA}, Memory quota: {_StarvationScenario.MEMORY_QUOTA}, "
            f"HWP CPU default: {_StarvationScenario.HWP_CPU_DEFAULT}, "
            f"HWP Memory default: {_StarvationScenario.HWP_MEMORY_DEFAULT}). "
            f"Running: {running_pods}, Gated: {gated_pods}"
        )
        assert running_pods == 0, f"No pods should be running when quota is exceeded. Running: {running_pods}"

        workload = _get_notebook_workload(
            client=admin_client,
            namespace=kueue_notebook_namespace.name,
            notebook_name=kueue_notebook.name,
        )
        assert workload, "Kueue should create a Workload object even when quota is insufficient"
        assert not _workload_is_admitted(workload), (
            f"Workload '{workload.name}' should NOT be admitted when quota is exceeded. "
            f"Conditions: {workload.instance.status.get('conditions', [])}"
        )
        LOGGER.info(
            "Notebook pod correctly gated by Kueue due to HWP resources exceeding quota "
            f"(quota CPU={_StarvationScenario.CPU_QUOTA}, memory={_StarvationScenario.MEMORY_QUOTA}, "
            f"HWP defaults CPU={_StarvationScenario.HWP_CPU_DEFAULT}, "
            f"memory={_StarvationScenario.HWP_MEMORY_DEFAULT}), "
            f"workload '{workload.name}' not admitted"
        )

    @pytest.mark.parametrize(
        _FIXTURE_NAMES,
        [pytest.param(*_normal_resources_params(), id="test_stop_start")],
        indirect=True,
    )
    def test_kueue_notebook_stop_start(
        self,
        admin_client: DynamicClient,
        unprivileged_client: DynamicClient,
        kueue_notebook_namespace: Namespace,
        kueue_notebook_pvc: PersistentVolumeClaim,
        kueue_resource_flavor: ResourceFlavor,
        kueue_cluster_queue: ClusterQueue,
        kueue_local_queue: LocalQueue,
        kueue_hardware_profile: HardwareProfile,
        kueue_notebook: Notebook,
    ) -> None:
        """Verify stop/start of a Kueue-managed workbench via HWP preserves scheduling.

        Given a running notebook managed by Kueue (via HardwareProfile scheduling),
        When the workbench is stopped via kubeflow-resource-stopped annotation,
        Then the pod is terminated.
        When the annotation is removed (workbench restarted),
        Then a new pod is created, Kueue re-admits the workload, and the pod
            carries full Kueue scheduling labels.
        """
        notebook_pod = Pod(
            client=unprivileged_client,
            namespace=kueue_notebook_namespace.name,
            name=f"{kueue_notebook.name}-0",
        )
        notebook_pod.wait(timeout=Timeout.TIMEOUT_2MIN)
        notebook_pod.wait_for_condition(
            condition=Pod.Condition.READY,
            status=Pod.Condition.Status.TRUE,
            timeout=Timeout.TIMEOUT_5MIN,
        )

        # Stop the workbench
        stop_timestamp = datetime.now(tz=UTC).strftime(format="%Y-%m-%dT%H:%M:%SZ")
        kueue_notebook.get()
        current_annotations = dict(kueue_notebook.instance.metadata.annotations or {})
        current_annotations[_KUBEFLOW_STOPPED_ANNOTATION] = stop_timestamp

        ResourceEditor(patches={kueue_notebook: {"metadata": {"annotations": current_annotations}}}).update()

        notebook_pod.wait_deleted(timeout=Timeout.TIMEOUT_2MIN)
        LOGGER.info(f"Notebook pod terminated after stop annotation (timestamp={stop_timestamp})")

        # Start the workbench by removing the stopped annotation (set to None for merge-patch deletion)
        kueue_notebook.get()
        ResourceEditor(
            patches={kueue_notebook: {"metadata": {"annotations": {_KUBEFLOW_STOPPED_ANNOTATION: None}}}}
        ).update()

        restarted_pod = Pod(
            client=unprivileged_client,
            namespace=kueue_notebook_namespace.name,
            name=f"{kueue_notebook.name}-0",
        )
        restarted_pod.wait(timeout=Timeout.TIMEOUT_2MIN)
        assert restarted_pod.exists, "New notebook pod should be created after restart"

        _assert_kueue_pod_labels(
            pod=restarted_pod,
            cluster_queue_name=kueue_cluster_queue.name,
            local_queue_name=kueue_local_queue.name,
        )

        restarted_pod.wait_for_condition(
            condition=Pod.Condition.READY,
            status=Pod.Condition.Status.TRUE,
            timeout=Timeout.TIMEOUT_5MIN,
        )

        workload = _get_notebook_workload(
            client=admin_client,
            namespace=kueue_notebook_namespace.name,
            notebook_name=kueue_notebook.name,
        )
        assert workload, f"Kueue should have a Workload object after restart for '{kueue_notebook.name}'"
        assert _workload_is_admitted(workload), (
            f"Workload '{workload.name}' should be re-admitted after restart. "
            f"Conditions: {workload.instance.status.get('conditions', [])}"
        )
        LOGGER.info(
            f"Notebook pod '{restarted_pod.name}' successfully restarted under Kueue management, "
            f"workload '{workload.name}' re-admitted"
        )
