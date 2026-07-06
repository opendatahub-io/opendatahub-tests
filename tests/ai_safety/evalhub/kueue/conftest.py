"""Fixtures for EvalHub Kueue integration tests.

Several helpers use subprocess(["oc", ...]) rather than ocp_resources DynamicClient.
This is intentional for two cases:
- Applying raw YAML for the Kueue CR (no ocp_resources wrapper exists)
- Force-patching finalizers on Kueue cluster-scoped resources stuck after controller removal
In all other cases, DynamicClient is preferred.
"""

import subprocess
from collections.abc import Generator
from contextlib import ExitStack
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.cluster_service_version import ClusterServiceVersion
from ocp_resources.custom_resource_definition import CustomResourceDefinition
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.deployment import Deployment
from ocp_resources.evalhub import EvalHub
from ocp_resources.namespace import Namespace
from ocp_resources.operator_group import OperatorGroup
from ocp_resources.resource import ResourceEditor
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.route import Route
from ocp_resources.service import Service
from ocp_resources.service_account import ServiceAccount
from ocp_resources.subscription import Subscription
from pytest_testconfig import config as py_config
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_safety.evalhub.constants import (
    EVALHUB_JOBS_WRITER_CLUSTERROLE,
    EVALHUB_TENANT_LABEL_KEY,
    EVALHUB_USER_ROLE_RULES,
    EVALHUB_VLLM_EMULATOR_PORT,
)
from tests.ai_safety.evalhub.kueue.constants import (
    MULTI_JOB_CPU_QUOTA,
    MULTI_JOB_MEMORY_QUOTA,
    SINGLE_JOB_CPU_QUOTA,
    SINGLE_JOB_MEMORY_QUOTA,
    VLLM_EMULATOR,
    VLLM_EMULATOR_IMAGE,
)
from tests.ai_safety.evalhub.utils import (
    build_evalhub_job_payload,
    delete_evalhub_job,
    submit_evalhub_job,
    tenant_rbac_ready,
)
from utilities.certificates_utils import create_ca_bundle_file
from utilities.constants import DscComponents, Labels, Protocols, Timeout
from utilities.data_science_cluster_utils import get_dsc_ready_condition, wait_for_dsc_reconciliation
from utilities.infra import create_inference_token, create_ns
from utilities.kueue_utils import (
    ClusterQueue,
    LocalQueue,
    ResourceFlavor,
    create_cluster_queue,
    create_local_queue,
    create_resource_flavor,
    wait_for_kueue_crds_available,
)

LOGGER = structlog.get_logger(name=__name__)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def _is_evalhub_crd_available(admin_client: DynamicClient) -> bool:
    """Check if EvalHub CRD is installed on the cluster."""
    crd_name = "evalhubs.trustyai.opendatahub.io"
    try:
        crd = CustomResourceDefinition(
            client=admin_client,
            name=crd_name,
        )
        return crd.exists
    except AttributeError, KeyError:
        return False


def _get_evalhub_class(admin_client: DynamicClient) -> type:
    """Return the EvalHub class matching the CRD version available on the cluster.

    The EvalHub CRD was promoted from v1alpha1 to v1 in a recent TrustyAI operator release.
    RHOAI 3.5.0-ea.2 and older ship v1alpha1; RHOAI 3.5.0 and newer ship v1.
    """
    try:
        crd = CustomResourceDefinition(
            client=admin_client,
            name="evalhubs.trustyai.opendatahub.io",
        )
        versions = [v["name"] for v in (crd.instance.spec.versions or [])]
    except ResourceNotFoundError, AttributeError:
        versions = []

    if "v1" in versions:
        return EvalHub

    # Cluster only has v1alpha1 — create a subclass with the correct api_version
    class EvalHubV1Alpha1(EvalHub):
        api_version: str = "trustyai.opendatahub.io/v1alpha1"

    return EvalHubV1Alpha1


# ---------------------------------------------------------------------------
# EvalHub Multi-Tenancy Fixtures (for Kueue tests)
# ---------------------------------------------------------------------------


# Kueue-specific evalhub_mt_* fixtures (use evalhub_kueue_namespace instead of model_namespace)
@pytest.fixture(scope="session")
def evalhub_kueue_cr(
    admin_client: DynamicClient,
    evalhub_kueue_namespace: Namespace,
) -> Generator[EvalHub, Any, Any]:
    """Create an EvalHub CR for Kueue tests."""
    if not _is_evalhub_crd_available(admin_client):
        pytest.fail(
            "EvalHub CRD 'evalhubs.trustyai.opendatahub.io' not available on this cluster. "
            "Install the TrustyAI/EvalHub operator first."
        )

    evalhub_cls = _get_evalhub_class(admin_client=admin_client)
    with evalhub_cls(
        client=admin_client,
        name="evalhub-mt",
        namespace=evalhub_kueue_namespace.name,
        database={"type": "sqlite"},
        collections=["leaderboard-v2"],
        wait_for_resource=True,
    ) as evalhub:
        yield evalhub


@pytest.fixture(scope="session")
def evalhub_kueue_deployment(
    admin_client: DynamicClient,
    evalhub_kueue_namespace: Namespace,
    evalhub_kueue_cr: EvalHub,
) -> Deployment:
    """Wait for the EvalHub deployment to become available.

    Note: The operator-generated kube-rbac-proxy auth.yaml only covers the
    collection-level path (/api/v1/evaluations/jobs) and not individual job
    paths (/api/v1/evaluations/jobs/{id}). Manual patching of the ConfigMap
    is not possible because the TrustyAI operator immediately reconciles it
    back to the original state. Tests that need per-job operations should use
    admin client K8s API calls instead of the EvalHub HTTP API.
    """
    deployment = Deployment(
        client=admin_client,
        name=evalhub_kueue_cr.name,
        namespace=evalhub_kueue_namespace.name,
    )
    deployment.wait_for_replicas(timeout=Timeout.TIMEOUT_5MIN)
    return deployment


@pytest.fixture(scope="session")
def evalhub_kueue_route(
    admin_client: DynamicClient,
    evalhub_kueue_namespace: Namespace,
    evalhub_kueue_deployment: Deployment,
) -> Route:
    """Get the Route for the EvalHub service."""
    return Route(
        client=admin_client,
        name=evalhub_kueue_deployment.name,
        namespace=evalhub_kueue_namespace.name,
        ensure_exists=True,
    )


@pytest.fixture(scope="session")
def evalhub_kueue_ca_bundle_file(
    admin_client: DynamicClient,
) -> str:
    """CA bundle file for verifying TLS on the EvalHub route."""
    return create_ca_bundle_file(client=admin_client)


# ---------------------------------------------------------------------------
# Kueue Fixtures
# ---------------------------------------------------------------------------


_KUEUE_OPERATOR_NS = "openshift-kueue-operator"
_KUEUE_PACKAGE = "kueue-operator"
_KUEUE_CHANNEL = "stable-v1.3"
_CERT_MANAGER_NS = "cert-manager-operator"
_CERT_MANAGER_PACKAGE = "openshift-cert-manager-operator"
_CERT_MANAGER_CHANNEL = "stable-v1"


def _is_kueue_operator_installed(admin_client: DynamicClient) -> bool:
    """Check if the Kueue operator is installed and ready in any namespace."""
    _kueue_namespaces = [
        py_config["applications_namespace"],
        _KUEUE_OPERATOR_NS,
        "openshift-operators",
    ]
    for namespace in _kueue_namespaces:
        try:
            csvs = list(ClusterServiceVersion.get(client=admin_client, namespace=namespace))
            for csv in csvs:
                if (
                    csv.instance.status.phase == "Succeeded"
                    and "kueue" in csv.instance.spec.get("displayName", "").lower()
                ):
                    LOGGER.info(f"Found Kueue operator CSV: {csv.name}")
                    return True
        except ResourceNotFoundError:
            continue
    return False


def _is_cert_manager_installed(admin_client: DynamicClient) -> bool:
    """Check if cert-manager is running (required by Kueue for webhook TLS)."""
    try:
        from ocp_resources.pod import Pod

        pods = list(Pod.get(client=admin_client, namespace="cert-manager", label_selector="app=cert-manager"))
        return any(pod.instance.status.phase == "Running" for pod in pods)
    except ResourceNotFoundError, AttributeError:
        return False


def _install_olm_operator(
    admin_client: DynamicClient,
    operator_ns: str,
    package: str,
    channel: str,
    source: str = "redhat-operators",
) -> None:
    """Install an OLM operator by creating Namespace, OperatorGroup, and Subscription."""
    ns = Namespace(client=admin_client, name=operator_ns)
    if not ns.exists:
        ns.create()
        LOGGER.info(f"Created namespace {operator_ns}")

    # OLM requires exactly one OperatorGroup per namespace — delete extras if present
    existing_ogs = list(OperatorGroup.get(client=admin_client, namespace=operator_ns))
    if len(existing_ogs) > 1:
        LOGGER.warning(f"Found {len(existing_ogs)} OperatorGroups in {operator_ns}, removing extras")
        for og in existing_ogs[1:]:
            og.delete(wait=True)
    elif len(existing_ogs) == 0:
        og = OperatorGroup(client=admin_client, name=package, namespace=operator_ns)
        og.create()
        LOGGER.info(f"Created OperatorGroup {package}")

    sub = Subscription(
        client=admin_client,
        name=package,
        namespace=operator_ns,
        channel=channel,
        source=source,
        source_namespace="openshift-marketplace",
        install_plan_approval="Automatic",
    )
    if not sub.exists:
        sub.create()
        LOGGER.info(f"Created Subscription {package}")


def _get_csvs(admin_client: DynamicClient, namespace: str) -> list:
    """Return all CSVs in a namespace."""
    try:
        return list(ClusterServiceVersion.get(client=admin_client, namespace=namespace))
    except ResourceNotFoundError:
        return []


def _wait_for_csv_succeeded(admin_client: DynamicClient, namespace: str, package: str, timeout: int = 600) -> None:
    """Wait for an OLM CSV to reach Succeeded phase.

    Matches on the CSV name containing the package name OR the package name
    containing a fragment of the CSV name (handles the openshift-cert-manager-operator
    package whose CSV is named cert-manager-operator.vX.Y.Z).
    """
    LOGGER.info(f"Waiting for {package} CSV to succeed in {namespace}")
    # Build candidate fragments — e.g. "openshift-cert-manager-operator" → also try "cert-manager-operator"
    fragments = [package, package.replace("openshift-", "")]
    for csvs in TimeoutSampler(
        wait_timeout=timeout,
        sleep=5,
        func=_get_csvs,
        admin_client=admin_client,
        namespace=namespace,
    ):
        for csv in csvs:
            if csv.instance.status.phase == "Succeeded" and any(f in csv.name for f in fragments):
                LOGGER.info(f"CSV {csv.name} succeeded")
                return


def _kueue_webhook_service_exists(admin_client: DynamicClient) -> bool:
    """Check if the Kueue webhook service exists."""
    return Service(
        client=admin_client,
        name="kueue-webhook-service",
        namespace=_KUEUE_OPERATOR_NS,
    ).exists


def _wait_for_kueue_webhook_service(admin_client: DynamicClient, timeout: int = 120) -> None:
    """Wait for the Kueue webhook service to be created by the controller."""
    LOGGER.info("Waiting for kueue-webhook-service to be created")
    for exists in TimeoutSampler(
        wait_timeout=timeout,
        sleep=10,
        func=_kueue_webhook_service_exists,
        admin_client=admin_client,
    ):
        if exists:
            LOGGER.info("kueue-webhook-service is ready")
            return


def _install_kueue(admin_client: DynamicClient) -> None:
    """Install cert-manager (if needed) and the Kueue operator, then create the Kueue CR."""
    # cert-manager is required by Kueue for webhook TLS
    if not _is_cert_manager_installed(admin_client):
        LOGGER.info("Installing cert-manager (required by Kueue)")
        _install_olm_operator(
            admin_client=admin_client,
            operator_ns=_CERT_MANAGER_NS,
            package=_CERT_MANAGER_PACKAGE,
            channel=_CERT_MANAGER_CHANNEL,
        )
        _wait_for_csv_succeeded(
            admin_client=admin_client,
            namespace=_CERT_MANAGER_NS,
            package=_CERT_MANAGER_PACKAGE,
        )
        # Wait for cert-manager pods to be running
        for _ in TimeoutSampler(
            wait_timeout=120,
            sleep=5,
            func=_is_cert_manager_installed,
            admin_client=admin_client,
        ):
            if _is_cert_manager_installed(admin_client):
                LOGGER.info("cert-manager is ready")
                break

    LOGGER.info("Installing Kueue operator")
    _install_olm_operator(
        admin_client=admin_client,
        operator_ns=_KUEUE_OPERATOR_NS,
        package=_KUEUE_PACKAGE,
        channel=_KUEUE_CHANNEL,
    )
    _wait_for_csv_succeeded(
        admin_client=admin_client,
        namespace=_KUEUE_OPERATOR_NS,
        package=_KUEUE_PACKAGE,
    )

    # Create the Kueue CR — the operator won't deploy the controller without it
    kueue_cr_yaml = (
        "apiVersion: kueue.openshift.io/v1\n"
        "kind: Kueue\n"
        "metadata:\n"
        "  name: cluster\n"
        "spec:\n"
        "  config:\n"
        "    integrations:\n"
        "      frameworks:\n"
        "      - BatchJob\n"
        "  managementState: Managed\n"
    )
    result = subprocess.run(
        args=["oc", "apply", "-f", "-"],
        input=kueue_cr_yaml,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode == 0:
        LOGGER.info("Applied Kueue CR 'cluster'")
    else:
        LOGGER.warning(f"Failed to apply Kueue CR: {result.stderr}")

    _wait_for_kueue_webhook_service(admin_client=admin_client)


def _force_delete_namespace(ns_name: str) -> None:
    """Force-delete a namespace stuck in Terminating by clearing its finalizers via oc.

    Kueue CRD removal causes stale visibility.kueue.x-k8s.io API group discovery
    entries which block normal namespace termination. This patches out the namespace
    spec.finalizers so the API server can complete deletion.
    """
    result = subprocess.run(
        args=[
            "oc",
            "get",
            "namespace",
            ns_name,
            "-o",
            "jsonpath={.spec.finalizers}",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if result.returncode != 0:
        return  # already gone
    patch_result = subprocess.run(
        args=[
            "oc",
            "patch",
            "namespace",
            ns_name,
            "--type=merge",
            "--patch",
            '{"spec":{"finalizers":[]}}',
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if patch_result.returncode == 0:
        LOGGER.info(f"Force-cleared finalizers on namespace {ns_name}")
    else:
        LOGGER.warning(f"Could not clear finalizers on {ns_name}: {patch_result.stderr}")


def _delete_namespace_with_force(ns_name: str, wait_seconds: int = 60) -> None:
    """Delete a namespace normally, then force-clear finalizers if it gets stuck."""
    result = subprocess.run(
        args=["oc", "delete", "namespace", ns_name, "--wait=true", f"--timeout={wait_seconds}s"],
        capture_output=True,
        text=True,
        timeout=wait_seconds + 15,
        check=False,
    )
    if result.returncode == 0:
        LOGGER.info(f"Deleted namespace {ns_name}")
    else:
        LOGGER.warning(f"Namespace {ns_name} deletion timed out, force-clearing finalizers")
        _force_delete_namespace(ns_name=ns_name)


def _uninstall_cert_manager(admin_client: DynamicClient) -> None:
    """Remove the cert-manager operator, its operator namespace, and the cert-manager workload namespace."""
    LOGGER.info("Uninstalling cert-manager")
    try:
        sub = Subscription(client=admin_client, name=_CERT_MANAGER_PACKAGE, namespace=_CERT_MANAGER_NS)
        if sub.exists:
            sub.delete()
    except Exception as e:  # noqa: BLE001  # teardown must not raise; log and continue
        LOGGER.warning(f"Failed to delete cert-manager subscription: {e}")

    # Delete both the operator namespace and the workload namespace
    for ns_name in (_CERT_MANAGER_NS, "cert-manager"):
        try:
            ns = Namespace(client=admin_client, name=ns_name)
            if ns.exists:
                ns.delete(wait=False)
                _delete_namespace_with_force(ns_name=ns_name)
        except Exception as e:  # noqa: BLE001  # teardown must not raise; log and continue
            LOGGER.warning(f"Failed to delete {ns_name} namespace: {e}")


def _force_clear_kueue_resource_finalizers() -> None:
    """Remove kueue.x-k8s.io/resource-in-use finalizers from ClusterQueues and ResourceFlavors.

    When Kueue is uninstalled, its controller stops running and can no longer
    remove these finalizers automatically. Force-clearing them allows the resources
    to be deleted so the test namespace and Kueue operator namespace can terminate.
    """
    for resource_type in ("clusterqueue", "resourceflavor"):
        list_result = subprocess.run(
            args=["oc", "get", resource_type, "--no-headers", "-o", "name"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if list_result.returncode != 0 or not list_result.stdout.strip():
            continue
        for resource_name in list_result.stdout.strip().splitlines():
            patch_result = subprocess.run(
                args=[
                    "oc",
                    "patch",
                    resource_name,
                    "--type=merge",
                    "--patch",
                    '{"metadata":{"finalizers":[]}}',
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if patch_result.returncode == 0:
                LOGGER.info(f"Cleared finalizers on {resource_name}")


def _uninstall_kueue(admin_client: DynamicClient) -> None:
    """Remove the Kueue operator and its namespace."""
    LOGGER.info("Uninstalling Kueue operator")
    # Clear Kueue resource finalizers first so they can be deleted once Kueue stops
    _force_clear_kueue_resource_finalizers()

    try:
        sub = Subscription(client=admin_client, name=_KUEUE_PACKAGE, namespace=_KUEUE_OPERATOR_NS)
        if sub.exists:
            sub.delete()
    except Exception as e:  # noqa: BLE001  # teardown must not raise; log and continue
        LOGGER.warning(f"Failed to delete Kueue subscription: {e}")

    try:
        ns = Namespace(client=admin_client, name=_KUEUE_OPERATOR_NS)
        if ns.exists:
            ns.delete(wait=False)
            _delete_namespace_with_force(ns_name=_KUEUE_OPERATOR_NS)
    except Exception as e:  # noqa: BLE001  # teardown must not raise; log and continue
        LOGGER.warning(f"Failed to delete Kueue namespace: {e}")


@pytest.fixture(scope="session")
def kueue_unmanaged_dsc(admin_client: DynamicClient, dsc_resource: DataScienceCluster) -> Generator[None, Any, Any]:
    """Ensure Kueue is available, installing it if necessary, and clean up afterwards.

    Session scope: OLM operator installation (~2-3 min) and DSC patching are expensive.
    Safe to share across test classes because no test modifies Kueue's managementState
    independently — ResourceEditor restores the original DSC state at session teardown.
    """
    kueue_was_installed_by_fixture = False
    cert_manager_was_installed_by_fixture = False

    if not _is_kueue_operator_installed(admin_client):
        LOGGER.info("Kueue operator not found — installing it for tests")
        # _install_kueue also installs cert-manager if missing; track both for cleanup
        if not _is_cert_manager_installed(admin_client):
            cert_manager_was_installed_by_fixture = True
        _install_kueue(admin_client=admin_client)
        kueue_was_installed_by_fixture = True
    else:
        LOGGER.info("Kueue operator already installed, ensuring cert-manager and Kueue CR are present")
        # The OLM operator may be installed but cert-manager and the Kueue CR might be
        # missing (e.g. on a fresh cluster where DSC had kueue=Removed). Without
        # cert-manager the Kueue controller cannot start; without the Kueue CR the
        # controller is never deployed at all.
        if not _is_cert_manager_installed(admin_client):
            LOGGER.info("cert-manager not found — installing (required by Kueue webhooks)")
            cert_manager_was_installed_by_fixture = True
            _install_olm_operator(
                admin_client=admin_client,
                operator_ns=_CERT_MANAGER_NS,
                package=_CERT_MANAGER_PACKAGE,
                channel=_CERT_MANAGER_CHANNEL,
            )
            _wait_for_csv_succeeded(
                admin_client=admin_client,
                namespace=_CERT_MANAGER_NS,
                package=_CERT_MANAGER_PACKAGE,
            )
        # Apply the Kueue CR idempotently so the controller is deployed
        kueue_cr_yaml = (
            "apiVersion: kueue.openshift.io/v1\n"
            "kind: Kueue\n"
            "metadata:\n"
            "  name: cluster\n"
            "spec:\n"
            "  config:\n"
            "    integrations:\n"
            "      frameworks:\n"
            "      - BatchJob\n"
            "  managementState: Managed\n"
        )
        result = subprocess.run(
            args=["oc", "apply", "-f", "-"],
            input=kueue_cr_yaml,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode == 0:
            LOGGER.info("Kueue CR 'cluster' applied")
        else:
            LOGGER.warning(f"Failed to apply Kueue CR: {result.stderr}")
        _wait_for_kueue_webhook_service(admin_client=admin_client)

    # Check current Kueue state in DSC
    try:
        kueue_management_state = dsc_resource.instance.spec.components[DscComponents.KUEUE].managementState
    except (AttributeError, KeyError) as e:
        pytest.fail(f"Kueue component not found in DSC: {e}")

    # States where Kueue is externally managed — no DSC patch needed.
    # Removed: RHOAI is not involved with Kueue; operator is installed standalone.
    # Unmanaged: RHOAI knows about Kueue but does not control it.
    # Patching to Unmanaged is only needed when RHOAI currently owns Kueue (Managed).
    _no_patch_states = {DscComponents.ManagementState.UNMANAGED, "Removed"}

    with ExitStack() as stack:
        if kueue_management_state in _no_patch_states:
            LOGGER.info(f"Kueue managementState is {kueue_management_state!r}, no DSC patch needed")
        else:
            LOGGER.info(f"Patching Kueue from {kueue_management_state} to Unmanaged")
            ready_condition = get_dsc_ready_condition(dsc=dsc_resource)
            pre_patch_time = ready_condition.get("lastTransitionTime") if ready_condition else None
            dsc_dict = {
                "spec": {
                    "components": {DscComponents.KUEUE: {"managementState": DscComponents.ManagementState.UNMANAGED}}
                }
            }
            stack.enter_context(cm=ResourceEditor(patches={dsc_resource: dsc_dict}))
            wait_for_dsc_reconciliation(dsc=dsc_resource, baseline_time=pre_patch_time)

        wait_for_kueue_crds_available(client=admin_client)
        yield

    if kueue_was_installed_by_fixture:
        _uninstall_kueue(admin_client=admin_client)
    if cert_manager_was_installed_by_fixture:
        _uninstall_cert_manager(admin_client=admin_client)


# ---------------------------------------------------------------------------
# Namespace and Queue Fixtures
# ---------------------------------------------------------------------------


# Kueue-specific namespace fixture
@pytest.fixture(scope="session")
def evalhub_kueue_namespace(
    admin_client: DynamicClient,
) -> Generator[Namespace, Any, Any]:
    """Namespace with both EvalHub tenant label and Kueue opt-in label."""
    with create_ns(
        admin_client=admin_client,
        name="test-evalhub-kueue",
        labels={
            EVALHUB_TENANT_LABEL_KEY: "true",
            # Red Hat Kueue operator opt-in label (upstream uses kueue.x-k8s.io/managed)
            "kueue.openshift.io/managed": "true",
        },
    ) as ns:
        yield ns


# Multi-job quota fixtures
@pytest.fixture(scope="session")
def evalhub_kueue_multi_job_resource_flavor(
    admin_client: DynamicClient,
    kueue_unmanaged_dsc: None,
) -> Generator[ResourceFlavor, Any, Any]:
    """ResourceFlavor for multi-job quota tests."""
    with create_resource_flavor(
        name="evalhub-multi-flavor",
        client=admin_client,
    ) as resource_flavor:
        yield resource_flavor


@pytest.fixture(scope="session")
def evalhub_kueue_multi_job_cluster_queue(
    admin_client: DynamicClient,
    evalhub_kueue_multi_job_resource_flavor: ResourceFlavor,
    kueue_unmanaged_dsc: None,
) -> Generator[ClusterQueue, Any, Any]:
    """ClusterQueue with quota for multiple EvalHub jobs."""
    resource_groups = [
        {
            "coveredResources": ["cpu", "memory"],
            "flavors": [
                {
                    "name": evalhub_kueue_multi_job_resource_flavor.name,
                    "resources": [
                        {"name": "cpu", "nominalQuota": MULTI_JOB_CPU_QUOTA},
                        {"name": "memory", "nominalQuota": MULTI_JOB_MEMORY_QUOTA},
                    ],
                }
            ],
        }
    ]

    with create_cluster_queue(
        name="evalhub-multi-cluster-queue",
        client=admin_client,
        resource_groups=resource_groups,
        namespace_selector={},
    ) as cluster_queue:
        yield cluster_queue


@pytest.fixture(scope="session")
def evalhub_kueue_multi_job_local_queue(
    admin_client: DynamicClient,
    evalhub_kueue_namespace: Namespace,
    evalhub_kueue_multi_job_cluster_queue: ClusterQueue,
    kueue_unmanaged_dsc: None,
) -> Generator[LocalQueue, Any, Any]:
    """LocalQueue for multi-job tests."""
    with create_local_queue(
        name="evalhub-local-queue-multi",
        namespace=evalhub_kueue_namespace.name,
        cluster_queue=evalhub_kueue_multi_job_cluster_queue.name,
        client=admin_client,
    ) as local_queue:
        yield local_queue


# Single-job quota fixtures (for quota exhaustion tests)
@pytest.fixture(scope="session")
def evalhub_kueue_single_job_resource_flavor(
    admin_client: DynamicClient,
    kueue_unmanaged_dsc: None,
) -> Generator[ResourceFlavor, Any, Any]:
    """ResourceFlavor for single-job quota tests."""
    with create_resource_flavor(
        name="evalhub-single-flavor",
        client=admin_client,
    ) as resource_flavor:
        yield resource_flavor


@pytest.fixture(scope="session")
def evalhub_kueue_single_job_cluster_queue(
    admin_client: DynamicClient,
    evalhub_kueue_single_job_resource_flavor: ResourceFlavor,
    kueue_unmanaged_dsc: None,
) -> Generator[ClusterQueue, Any, Any]:
    """ClusterQueue with quota for exactly 1 EvalHub job."""
    resource_groups = [
        {
            "coveredResources": ["cpu", "memory"],
            "flavors": [
                {
                    "name": evalhub_kueue_single_job_resource_flavor.name,
                    "resources": [
                        {"name": "cpu", "nominalQuota": SINGLE_JOB_CPU_QUOTA},
                        {"name": "memory", "nominalQuota": SINGLE_JOB_MEMORY_QUOTA},
                    ],
                }
            ],
        }
    ]

    with create_cluster_queue(
        name="evalhub-single-cluster-queue",
        client=admin_client,
        resource_groups=resource_groups,
        namespace_selector={},
    ) as cluster_queue:
        yield cluster_queue


@pytest.fixture(scope="session")
def evalhub_kueue_single_job_local_queue(
    admin_client: DynamicClient,
    evalhub_kueue_namespace: Namespace,
    evalhub_kueue_single_job_cluster_queue: ClusterQueue,
    kueue_unmanaged_dsc: None,
) -> Generator[LocalQueue, Any, Any]:
    """LocalQueue in the EvalHub namespace for single-job tests."""
    with create_local_queue(
        name="evalhub-local-queue",
        namespace=evalhub_kueue_namespace.name,
        cluster_queue=evalhub_kueue_single_job_cluster_queue.name,
        client=admin_client,
    ) as local_queue:
        yield local_queue


# RBAC fixtures
@pytest.fixture(scope="session")
def evalhub_kueue_tenant_rbac(
    admin_client: DynamicClient,
    evalhub_kueue_namespace: Namespace,
    evalhub_kueue_deployment: Deployment,
) -> None:
    """Wait for operator to provision tenant RBAC in Kueue namespace."""
    try:
        for ready in TimeoutSampler(
            wait_timeout=120,
            sleep=5,
            func=tenant_rbac_ready,
            admin_client=admin_client,
            namespace=evalhub_kueue_namespace.name,
        ):
            if ready:
                LOGGER.info(f"Operator RBAC provisioned in {evalhub_kueue_namespace.name}")
                return
    except TimeoutExpiredError as exc:
        raise RuntimeError(f"Operator RBAC not provisioned in {evalhub_kueue_namespace.name} within 120s") from exc


# vLLM emulator in Kueue namespace
@pytest.fixture(scope="session")
def evalhub_kueue_vllm_emulator_deployment(
    admin_client: DynamicClient,
    evalhub_kueue_namespace: Namespace,
    evalhub_kueue_tenant_rbac: None,
) -> Generator[Deployment, Any, Any]:
    """Deploy vLLM emulator in the Kueue namespace."""
    label = {Labels.Openshift.APP: VLLM_EMULATOR}
    with Deployment(
        client=admin_client,
        namespace=evalhub_kueue_namespace.name,
        name=VLLM_EMULATOR,
        label=label,
        selector={"matchLabels": label},
        template={
            "metadata": {"labels": label, "name": VLLM_EMULATOR},
            "spec": {
                "containers": [
                    {
                        "image": VLLM_EMULATOR_IMAGE,
                        "name": VLLM_EMULATOR,
                        "ports": [{"containerPort": EVALHUB_VLLM_EMULATOR_PORT, "protocol": Protocols.TCP}],
                        "readinessProbe": {
                            "tcpSocket": {"port": EVALHUB_VLLM_EMULATOR_PORT},
                            "initialDelaySeconds": 5,
                            "periodSeconds": 5,
                        },
                        "securityContext": {
                            "allowPrivilegeEscalation": False,
                            "runAsNonRoot": True,
                            "capabilities": {"drop": ["ALL"]},
                            "seccompProfile": {"type": "RuntimeDefault"},
                        },
                    }
                ]
            },
        },
        replicas=1,
    ) as deployment:
        deployment.wait_for_replicas(timeout=Timeout.TIMEOUT_5MIN)
        yield deployment


@pytest.fixture(scope="session")
def evalhub_kueue_vllm_service(
    admin_client: DynamicClient,
    evalhub_kueue_namespace: Namespace,
    evalhub_kueue_vllm_emulator_deployment: Deployment,
) -> Generator[Service, Any, Any]:
    """Service for vLLM emulator."""
    with Service(
        client=admin_client,
        namespace=evalhub_kueue_namespace.name,
        name=f"{VLLM_EMULATOR}-service",
        ports=[
            {
                "name": f"{VLLM_EMULATOR}-endpoint",
                "port": EVALHUB_VLLM_EMULATOR_PORT,
                "protocol": Protocols.TCP,
                "targetPort": EVALHUB_VLLM_EMULATOR_PORT,
            }
        ],
        selector={Labels.Openshift.APP: VLLM_EMULATOR},
    ) as service:
        yield service


# User token fixture for API access
@pytest.fixture(scope="session")
def evalhub_kueue_user_token(
    admin_client: DynamicClient,
    evalhub_kueue_namespace: Namespace,
) -> str:
    """Create ServiceAccount and token for EvalHub API access."""
    with (
        ServiceAccount(
            client=admin_client,
            name="evalhub-kueue-user",
            namespace=evalhub_kueue_namespace.name,
            wait_for_resource=True,
        ) as sa,
        Role(
            client=admin_client,
            name="evalhub-kueue-user-role",
            namespace=evalhub_kueue_namespace.name,
            rules=EVALHUB_USER_ROLE_RULES,
            wait_for_resource=True,
        ) as role,
        RoleBinding(
            client=admin_client,
            name="evalhub-kueue-user-binding",
            namespace=evalhub_kueue_namespace.name,
            subjects_kind="ServiceAccount",
            subjects_name=sa.name,
            subjects_namespace=evalhub_kueue_namespace.name,
            role_ref_kind="Role",
            role_ref_name=role.name,
            wait_for_resource=True,
        ),
        # kube-rbac-proxy maps HTTP DELETE on /evaluations/jobs to delete on batch/jobs.
        # Bind the SA to the ClusterRole that grants this permission.
        RoleBinding(
            client=admin_client,
            name="evalhub-kueue-user-jobs-writer-binding",
            namespace=evalhub_kueue_namespace.name,
            subjects_kind="ServiceAccount",
            subjects_name=sa.name,
            subjects_namespace=evalhub_kueue_namespace.name,
            role_ref_kind="ClusterRole",
            role_ref_name=EVALHUB_JOBS_WRITER_CLUSTERROLE,
            wait_for_resource=True,
        ),
    ):
        yield create_inference_token(model_service_account=sa)


# ---------------------------------------------------------------------------
# Negative Test Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def evalhub_job_with_nonexistent_queue(
    evalhub_kueue_namespace: Namespace,
    evalhub_kueue_vllm_service: Service,
    evalhub_kueue_route: Route,
    evalhub_kueue_user_token: str,
    evalhub_kueue_ca_bundle_file: str,
):
    """Fixture that submits a job with non-existent queue and ensures cleanup."""
    payload = build_evalhub_job_payload(
        model_service_name=evalhub_kueue_vllm_service.name,
        tenant_namespace=evalhub_kueue_namespace.name,
        job_name="tc-neg-001-invalid-queue",
    )
    payload["queue"] = {"kind": "kueue", "name": "nonexistent-queue"}

    data = submit_evalhub_job(
        host=evalhub_kueue_route.host,
        token=evalhub_kueue_user_token,
        ca_bundle_file=evalhub_kueue_ca_bundle_file,
        tenant=evalhub_kueue_namespace.name,
        payload=payload,
    )
    job_id = data["resource"]["id"]

    yield {
        "job_id": job_id,
        "host": evalhub_kueue_route.host,
        "token": evalhub_kueue_user_token,
        "ca_bundle_file": evalhub_kueue_ca_bundle_file,
        "tenant": evalhub_kueue_namespace.name,
    }

    # Cleanup - always executes even if test fails
    delete_evalhub_job(
        host=evalhub_kueue_route.host,
        token=evalhub_kueue_user_token,
        ca_bundle_file=evalhub_kueue_ca_bundle_file,
        tenant=evalhub_kueue_namespace.name,
        job_id=job_id,
        hard_delete=True,
    )


@pytest.fixture
def evalhub_job_without_queue_spec(
    evalhub_kueue_namespace: Namespace,
    evalhub_kueue_vllm_service: Service,
    evalhub_kueue_route: Route,
    evalhub_kueue_user_token: str,
    evalhub_kueue_ca_bundle_file: str,
):
    """Fixture that submits a job without queue spec and ensures cleanup."""
    payload = build_evalhub_job_payload(
        model_service_name=evalhub_kueue_vllm_service.name,
        tenant_namespace=evalhub_kueue_namespace.name,
        job_name="tc-neg-002-no-queue",
    )
    payload.pop("queue", None)

    data = submit_evalhub_job(
        host=evalhub_kueue_route.host,
        token=evalhub_kueue_user_token,
        ca_bundle_file=evalhub_kueue_ca_bundle_file,
        tenant=evalhub_kueue_namespace.name,
        payload=payload,
    )
    job_id = data["resource"]["id"]

    yield {
        "job_id": job_id,
        "host": evalhub_kueue_route.host,
        "token": evalhub_kueue_user_token,
        "ca_bundle_file": evalhub_kueue_ca_bundle_file,
        "tenant": evalhub_kueue_namespace.name,
    }

    # Cleanup - always executes even if test fails
    delete_evalhub_job(
        host=evalhub_kueue_route.host,
        token=evalhub_kueue_user_token,
        ca_bundle_file=evalhub_kueue_ca_bundle_file,
        tenant=evalhub_kueue_namespace.name,
        job_id=job_id,
        hard_delete=True,
    )


@pytest.fixture
def evalhub_job_for_cross_tenant_test(
    evalhub_kueue_namespace: Namespace,
    evalhub_kueue_multi_job_local_queue: LocalQueue,
    evalhub_kueue_vllm_service: Service,
    evalhub_kueue_route: Route,
    evalhub_kueue_user_token: str,
    evalhub_kueue_ca_bundle_file: str,
):
    """Fixture that submits a job for cross-tenant access testing and ensures cleanup."""
    payload = build_evalhub_job_payload(
        model_service_name=evalhub_kueue_vllm_service.name,
        tenant_namespace=evalhub_kueue_namespace.name,
        job_name="tc-neg-004-cross-tenant",
    )
    payload["queue"] = {"kind": "kueue", "name": evalhub_kueue_multi_job_local_queue.name}

    data = submit_evalhub_job(
        host=evalhub_kueue_route.host,
        token=evalhub_kueue_user_token,
        ca_bundle_file=evalhub_kueue_ca_bundle_file,
        tenant=evalhub_kueue_namespace.name,
        payload=payload,
    )
    job_id = data["resource"]["id"]

    yield {
        "job_id": job_id,
        "host": evalhub_kueue_route.host,
        "token": evalhub_kueue_user_token,
        "ca_bundle_file": evalhub_kueue_ca_bundle_file,
        "tenant": evalhub_kueue_namespace.name,
    }

    # Cleanup - always executes even if test fails
    delete_evalhub_job(
        host=evalhub_kueue_route.host,
        token=evalhub_kueue_user_token,
        ca_bundle_file=evalhub_kueue_ca_bundle_file,
        tenant=evalhub_kueue_namespace.name,
        job_id=job_id,
        hard_delete=True,
    )
