from collections.abc import Generator
from contextlib import ExitStack
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.cluster_service_version import ClusterServiceVersion
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.resource import ResourceEditor
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.service import Service
from ocp_resources.service_account import ServiceAccount
from pytest_testconfig import config as py_config
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.model_explainability.evalhub.constants import (
    EVALHUB_TENANT_LABEL_KEY,
    EVALHUB_USER_ROLE_RULES,
    EVALHUB_VLLM_EMULATOR_PORT,
)
from tests.model_explainability.evalhub.kueue.constants import (
    MULTI_JOB_CPU_QUOTA,
    MULTI_JOB_MEMORY_QUOTA,
    SINGLE_JOB_CPU_QUOTA,
    SINGLE_JOB_MEMORY_QUOTA,
    VLLM_EMULATOR,
    VLLM_EMULATOR_IMAGE,
)
from tests.model_explainability.evalhub.utils import tenant_rbac_ready
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
            if csv.name.startswith("kueue") and csv.status == csv.Status.SUCCEEDED:
                LOGGER.info(f"Found Kueue operator CSV: {csv.name}")
                return True
        return False
    except ResourceNotFoundError:
        return False


@pytest.fixture(scope="session")
def kueue_unmanaged_dsc(admin_client: DynamicClient, dsc_resource: DataScienceCluster) -> Generator[None, Any, Any]:
    """Set DSC Kueue to Unmanaged and wait for CRDs to be available."""
    try:
        if not _is_kueue_operator_installed(admin_client):
            pytest.fail("Kueue operator is not installed")

        # Check current Kueue state
        kueue_management_state = dsc_resource.instance.spec.components[DscComponents.KUEUE].managementState

        with ExitStack() as stack:
            # Only patch if Kueue is not already Unmanaged
            if kueue_management_state != DscComponents.ManagementState.UNMANAGED:
                LOGGER.info(f"Patching Kueue from {kueue_management_state} to Unmanaged")
                # Read timestamp BEFORE applying patch
                ready_condition = get_dsc_ready_condition(dsc=dsc_resource)
                pre_patch_time = ready_condition.get("lastTransitionTime") if ready_condition else None

                dsc_dict = {
                    "spec": {
                        "components": {
                            DscComponents.KUEUE: {"managementState": DscComponents.ManagementState.UNMANAGED}
                        }
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

    except (AttributeError, KeyError) as e:
        pytest.fail(f"Kueue component not found in DSC: {e}")


# ---------------------------------------------------------------------------
# Namespace and Queue Fixtures
# ---------------------------------------------------------------------------


# Kueue-specific namespace fixture
@pytest.fixture(scope="class")
def evalhub_kueue_namespace(
    admin_client: DynamicClient,
) -> Generator[Namespace, Any, Any]:
    """Namespace with both EvalHub tenant label and Kueue queue label."""
    with create_ns(
        admin_client=admin_client,
        name="test-evalhub-kueue",
        labels={
            EVALHUB_TENANT_LABEL_KEY: "true",
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
    kueue_unmanaged_dsc,
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
    kueue_unmanaged_dsc,
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
    kueue_unmanaged_dsc,
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
def evalhub_kueue_rbac_ready(
    admin_client: DynamicClient,
    evalhub_kueue_namespace: Namespace,
    evalhub_mt_deployment: Deployment,
) -> None:
    """Wait for operator to provision tenant RBAC in Kueue namespace."""
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


# vLLM emulator in Kueue namespace
@pytest.fixture(scope="class")
def evalhub_kueue_vllm_emulator_deployment(
    admin_client: DynamicClient,
    evalhub_kueue_namespace: Namespace,
    evalhub_kueue_rbac_ready: None,
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
