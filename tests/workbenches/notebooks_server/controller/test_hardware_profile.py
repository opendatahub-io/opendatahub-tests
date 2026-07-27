from collections.abc import Generator
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.notebook import Notebook
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from timeout_sampler import TimeoutExpiredError

from tests.workbenches.notebooks_server.controller.utils import HardwareProfile, build_notebook_dict
from utilities.constants import Timeout
from utilities.infra import create_ns

LOGGER = structlog.get_logger(name=__name__)

HWP_ANNOTATION_NAME = "opendatahub.io/hardware-profile-name"
HWP_ANNOTATION_NAMESPACE = "opendatahub.io/hardware-profile-namespace"


@pytest.fixture()
def hwp_remote_namespace(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
) -> Generator[Namespace | None]:
    """Optionally creates a separate namespace for a cross-namespace HardwareProfile.

    If params contain a "name" key, creates and yields the namespace.
    If params are empty or not provided, yields None (same-namespace scenario).
    """
    params = getattr(request, "param", None) or {}
    ns_name = params.get("name")
    if not ns_name:
        yield None
        return

    with create_ns(admin_client=admin_client, name=ns_name) as ns:
        yield ns


@pytest.fixture()
def hardware_profile(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    hwp_remote_namespace: Namespace | None,
) -> Generator[HardwareProfile]:
    """Creates a HardwareProfile CR.

    If hwp_remote_namespace is provided (cross-namespace scenario), the profile
    is created there. Otherwise it is created in the notebook's namespace.
    """
    params = request.param
    profile_name = params["name"]
    identifiers = params["identifiers"]

    target_namespace = hwp_remote_namespace.name if hwp_remote_namespace else unprivileged_model_namespace.name

    kind_dict: dict[str, Any] = {
        "apiVersion": "infrastructure.opendatahub.io/v1",
        "kind": "HardwareProfile",
        "metadata": {
            "name": profile_name,
            "namespace": target_namespace,
        },
        "spec": {
            "identifiers": identifiers,
        },
    }

    with HardwareProfile(client=admin_client, kind_dict=kind_dict) as hwp:
        yield hwp


@pytest.fixture()
def hwp_notebook(
    request: pytest.FixtureRequest,
    unprivileged_client: DynamicClient,
    notebook_image: str,
    users_persistent_volume_claim: PersistentVolumeClaim,
    hardware_profile: HardwareProfile,
) -> Generator[Notebook]:
    """Creates a Notebook CR after the HardwareProfile exists.

    Depends on hardware_profile to guarantee the profile is created before
    the Notebook, so the admission webhook can resolve it.
    """
    namespace = request.param["namespace"]
    name = request.param["name"]
    extra_annotations = request.param.get("extra_annotations", {})

    notebook_dict = build_notebook_dict(
        namespace=namespace,
        name=name,
        image_path=notebook_image,
        extra_annotations=extra_annotations or None,
        resources={},
    )

    with Notebook(client=unprivileged_client, kind_dict=notebook_dict) as nb:
        yield nb


@pytest.fixture()
def hwp_notebook_pod(
    request: pytest.FixtureRequest,
    unprivileged_client: DynamicClient,
    hwp_notebook: Notebook,
) -> Pod:
    """Returns the notebook pod in Ready state (depends on hwp_notebook)."""
    params = getattr(request, "param", {})
    pod_ready_timeout = params.get("timeout", 600)

    pod = Pod(
        client=unprivileged_client,
        namespace=hwp_notebook.namespace,
        name=f"{hwp_notebook.name}-0",
    )

    try:
        pod.wait(timeout=pod_ready_timeout)
        pod.wait_for_condition(
            condition=Pod.Condition.READY,
            status=Pod.Condition.Status.TRUE,
            timeout=pod_ready_timeout,
        )
    except (TimeoutError, TimeoutExpiredError) as timeout_err:
        try:
            pod_exists = pod.exists
        except Exception:  # noqa: BLE001
            LOGGER.warning("Failed to verify pod existence after timeout")
            pod_exists = False

        if pod_exists:
            raise AssertionError(
                f"Pod '{hwp_notebook.name}-0' failed to reach Ready state within {pod_ready_timeout} seconds."
            ) from timeout_err
        else:
            raise AssertionError(
                f"Pod '{hwp_notebook.name}-0' was not created. Check notebook controller logs."
            ) from timeout_err

    return pod


_HWP_IDENTIFIERS = (
    {
        "displayName": "CPU",
        "identifier": "cpu",
        "minCount": "100m",
        "maxCount": "400m",
        "defaultCount": "200m",
        "resourceType": "CPU",
    },
    {
        "displayName": "Memory",
        "identifier": "memory",
        "minCount": "128Mi",
        "maxCount": "512Mi",
        "defaultCount": "256Mi",
        "resourceType": "Memory",
    },
)


class TestHardwareProfileIntegration:
    """Verify that HardwareProfile webhook injects resources into Notebook pods."""

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        "unprivileged_model_namespace,users_persistent_volume_claim,"
        "hwp_remote_namespace,hardware_profile,hwp_notebook,hwp_notebook_pod",
        [
            pytest.param(
                {
                    "name": "test-nb-hwp",
                    "add-dashboard-label": True,
                },
                {"name": "test-nb-hwp"},
                {},
                {
                    "name": "test-hwp-cpu-mem",
                    "identifiers": _HWP_IDENTIFIERS,
                },
                {
                    "namespace": "test-nb-hwp",
                    "name": "test-nb-hwp",
                    "extra_annotations": {
                        HWP_ANNOTATION_NAME: "test-hwp-cpu-mem",
                    },
                },
                {"timeout": Timeout.TIMEOUT_2MIN},
                id="test_hwp_same_namespace",
            ),
            pytest.param(
                {
                    "name": "test-nb-hwp-xns",
                    "add-dashboard-label": True,
                },
                {"name": "test-nb-hwp-xns"},
                {"name": "test-hwp-shared"},
                {
                    "name": "test-hwp-cpu-mem-xns",
                    "identifiers": _HWP_IDENTIFIERS,
                },
                {
                    "namespace": "test-nb-hwp-xns",
                    "name": "test-nb-hwp-xns",
                    "extra_annotations": {
                        HWP_ANNOTATION_NAME: "test-hwp-cpu-mem-xns",
                        HWP_ANNOTATION_NAMESPACE: "test-hwp-shared",
                    },
                },
                {"timeout": Timeout.TIMEOUT_2MIN},
                id="test_hwp_cross_namespace",
            ),
        ],
        indirect=True,
    )
    def test_hardware_profile_injects_resources(
        self,
        hwp_notebook_pod: Pod,
        unprivileged_model_namespace: Namespace,
        users_persistent_volume_claim: PersistentVolumeClaim,
        hwp_remote_namespace: Namespace | None,
        hardware_profile: HardwareProfile,
        hwp_notebook: Notebook,
    ) -> None:
        """Verify HardwareProfile webhook injects resource requests/limits into the notebook pod.

        Given a HardwareProfile CR exists with CPU and Memory identifiers
            (either in the same namespace or a remote namespace),
        When a Notebook CR is created with the hardware-profile-name annotation
            (and hardware-profile-namespace for cross-namespace scenarios)
            and no explicit container resources,
        Then the webhook should inject defaultCount as both requests and limits (Guaranteed QoS).
        """
        notebook_container = _find_notebook_container(pod=hwp_notebook_pod, notebook_name=hwp_notebook.name)
        assert notebook_container, (
            f"Notebook container '{hwp_notebook.name}' not found in pod. "
            f"Available: {[container.name for container in hwp_notebook_pod.instance.spec.containers]}"
        )

        actual_requests = dict(notebook_container.resources.requests or {})
        actual_limits = dict(notebook_container.resources.limits or {})

        # The webhook sets both requests and limits to defaultCount (Guaranteed QoS).
        # MaxCount is only a UI-side validation ceiling and does not flow into pod specs.
        expected_cpu = "200m"
        expected_memory = "256Mi"

        assert actual_requests.get("cpu") == expected_cpu, (
            f"CPU request mismatch: expected '{expected_cpu}', got '{actual_requests.get('cpu')}'"
        )
        assert actual_requests.get("memory") == expected_memory, (
            f"Memory request mismatch: expected '{expected_memory}', got '{actual_requests.get('memory')}'"
        )
        assert actual_limits.get("cpu") == expected_cpu, (
            f"CPU limit mismatch: expected '{expected_cpu}', got '{actual_limits.get('cpu')}'"
        )
        assert actual_limits.get("memory") == expected_memory, (
            f"Memory limit mismatch: expected '{expected_memory}', got '{actual_limits.get('memory')}'"
        )


def _find_notebook_container(pod: Pod, notebook_name: str) -> Any | None:
    """Find the notebook container in the pod by matching the notebook name."""
    for container in pod.instance.spec.containers:
        if container.name == notebook_name:
            return container
    return None
