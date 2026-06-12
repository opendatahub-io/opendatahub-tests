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
from ocp_resources.resource import ResourceEditor
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.route import Route
from ocp_resources.service import Service
from ocp_resources.service_account import ServiceAccount
from pytest_testconfig import config as py_config
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_safety.evalhub.constants import (
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


# ---------------------------------------------------------------------------
# EvalHub Multi-Tenancy Fixtures (for Kueue tests)
# ---------------------------------------------------------------------------


# Kueue-specific evalhub_mt_* fixtures (use evalhub_kueue_namespace instead of model_namespace)
@pytest.fixture(scope="class")
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

    with EvalHub(
        client=admin_client,
        name="evalhub-mt",
        namespace=evalhub_kueue_namespace.name,
        database={"type": "sqlite"},
        collections=["leaderboard-v2"],
        wait_for_resource=True,
    ) as evalhub:
        yield evalhub


@pytest.fixture(scope="class")
def evalhub_kueue_deployment(
    admin_client: DynamicClient,
    evalhub_kueue_namespace: Namespace,
    evalhub_kueue_cr: EvalHub,
) -> Deployment:
    """Wait for the EvalHub deployment to become available."""
    deployment = Deployment(
        client=admin_client,
        name=evalhub_kueue_cr.name,
        namespace=evalhub_kueue_namespace.name,
    )
    deployment.wait_for_replicas(timeout=Timeout.TIMEOUT_5MIN)
    return deployment


@pytest.fixture(scope="class")
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


@pytest.fixture(scope="class")
def evalhub_kueue_ca_bundle_file(
    admin_client: DynamicClient,
) -> str:
    """CA bundle file for verifying TLS on the EvalHub route."""
    return create_ca_bundle_file(client=admin_client)


# ---------------------------------------------------------------------------
# Kueue Fixtures
# ---------------------------------------------------------------------------


def _is_kueue_operator_installed(admin_client: DynamicClient) -> bool:
    """Check if the Kueue operator is installed and ready."""
    try:
        csvs = list(
            ClusterServiceVersion.get(
                client=admin_client,
                namespace=py_config["applications_namespace"],
            )
        )
        for csv in csvs:
            # Check phase instead of status attribute, and verify package name precisely
            if csv.instance.status.phase == "Succeeded" and "kueue" in csv.instance.spec.get("displayName", "").lower():
                LOGGER.info(f"Found Kueue operator CSV: {csv.name}")
                return True
        return False
    except ResourceNotFoundError:
        return False


@pytest.fixture(scope="session")
def kueue_unmanaged_dsc(admin_client: DynamicClient, dsc_resource: DataScienceCluster) -> Generator[None, Any, Any]:
    """Set DSC Kueue to Unmanaged and wait for CRDs to be available."""
    if not _is_kueue_operator_installed(admin_client):
        pytest.fail("Kueue operator is not installed")

    # Check current Kueue state - narrow try/except to component lookup only
    try:
        kueue_management_state = dsc_resource.instance.spec.components[DscComponents.KUEUE].managementState
    except (AttributeError, KeyError) as e:
        pytest.fail(f"Kueue component not found in DSC: {e}")

    with ExitStack() as stack:
        # Only patch if Kueue is not already Unmanaged
        if kueue_management_state != DscComponents.ManagementState.UNMANAGED:
            LOGGER.info(f"Patching Kueue from {kueue_management_state} to Unmanaged")
            # Read timestamp BEFORE applying patch
            ready_condition = get_dsc_ready_condition(dsc=dsc_resource)
            pre_patch_time = ready_condition.get("lastTransitionTime") if ready_condition else None

            dsc_dict = {
                "spec": {
                    "components": {DscComponents.KUEUE: {"managementState": DscComponents.ManagementState.UNMANAGED}}
                }
            }
            stack.enter_context(cm=ResourceEditor(patches={dsc_resource: dsc_dict}))

            # Wait for DSC to reconcile the patch
            wait_for_dsc_reconciliation(dsc=dsc_resource, baseline_time=pre_patch_time)
        else:
            LOGGER.info("Kueue already Unmanaged, no patch needed")

        # Always wait for Kueue CRDs and controller pods (regardless of patch)
        wait_for_kueue_crds_available(client=admin_client)
        yield


# ---------------------------------------------------------------------------
# Namespace and Queue Fixtures
# ---------------------------------------------------------------------------


# Kueue-specific namespace fixture
@pytest.fixture(scope="class")
def evalhub_kueue_namespace(
    admin_client: DynamicClient,
) -> Generator[Namespace, Any, Any]:
    """Namespace with both EvalHub tenant label and Kueue opt-in label."""
    with create_ns(
        admin_client=admin_client,
        name="test-evalhub-kueue",
        labels={
            EVALHUB_TENANT_LABEL_KEY: "true",
            "kueue.x-k8s.io/managed": "true",  # Required for Kueue admission control
        },
    ) as ns:
        yield ns


# Multi-job quota fixtures
@pytest.fixture(scope="class")
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


@pytest.fixture(scope="class")
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


@pytest.fixture(scope="class")
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
@pytest.fixture(scope="class")
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


@pytest.fixture(scope="class")
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


@pytest.fixture(scope="class")
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
@pytest.fixture(scope="class")
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
@pytest.fixture(scope="class")
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


@pytest.fixture(scope="class")
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
@pytest.fixture(scope="class")
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
