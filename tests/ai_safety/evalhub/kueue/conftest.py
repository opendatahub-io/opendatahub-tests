import time
from collections.abc import Generator
from contextlib import ExitStack
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.deployment import Deployment
from ocp_resources.evalhub import EvalHub
from ocp_resources.exceptions import ResourceTeardownError
from ocp_resources.namespace import Namespace
from ocp_resources.resource import ResourceEditor
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.route import Route
from ocp_resources.service import Service
from ocp_resources.service_account import ServiceAccount
from ocp_resources.subscription import Subscription
from ocp_utilities.operators import install_operator, uninstall_operator
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_safety.evalhub.constants import (
    EVALHUB_JOBS_WRITER_CLUSTERROLE,
    EVALHUB_TENANT_LABEL_KEY,
    EVALHUB_USER_ROLE_RULES,
    EVALHUB_VLLM_EMULATOR_PORT,
    VLLM_EMULATOR_IMAGE,
)
from tests.ai_safety.evalhub.kueue.constants import (
    KUEUE_CPU_QUOTA,
    KUEUE_MEMORY_QUOTA,
    VLLM_EMULATOR,
)
from tests.ai_safety.evalhub.utils import (
    build_evalhub_job_payload,
    build_evalhub_kueue_job_payload,
    cleanup_evalhub_job,
    is_evalhub_crd_available,
    submit_evalhub_job,
    tenant_rbac_ready,
)
from utilities.certificates_utils import create_ca_bundle_file
from utilities.constants import DscComponents, Labels, Protocols, Timeout
from utilities.data_science_cluster_utils import get_dsc_ready_condition, wait_for_dsc_reconciliation
from utilities.infra import create_inference_token, create_ns
from utilities.kueue_utils import (
    KUEUE_OPERATOR_NAMESPACE,
    ClusterQueue,
    Kueue,
    LocalQueue,
    ResourceFlavor,
    create_cluster_queue,
    create_local_queue,
    create_resource_flavor,
    drain_namespace_kueue_resources,
    full_kueue_controller_cleanup,
    get_kueue_controller_pod_uids,
    pause_kueue_controller,
    remove_kueue_visibility_api_services,
    resume_kueue_controller,
    wait_for_kueue_controller_rollout,
    wait_for_kueue_crds_available,
    wait_for_queue_active,
)

LOGGER = structlog.get_logger(name=__name__)

KUEUE_TENANT_NS = "test-evalhub-kueue"
KUEUE_MODEL_NS = "test-evalhub-kueue-model"
MULTI_JOB_FLAVOR_NAME = "evalhub-multi-flavor"
SINGLE_JOB_FLAVOR_NAME = "evalhub-single-flavor"
MULTI_JOB_CLUSTER_QUEUE_NAME = "evalhub-multi-cluster-queue"
SINGLE_JOB_CLUSTER_QUEUE_NAME = "evalhub-single-cluster-queue"


# ---------------------------------------------------------------------------
# EvalHub Multi-Tenancy Fixtures (for Kueue tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def trustyai_pods_log_rbac(
    admin_client: DynamicClient,
    evalhub_kueue_model_namespace: Namespace,
    evalhub_kueue_namespace: Namespace,
) -> Generator[None, Any, Any]:
    """Give the TrustyAI operator permission to read pod logs in test namespaces.

    In RHOAI 3.5.0, TrustyAI creates an EvalHub Role that includes permission
    to read pod logs, but the TrustyAI operator itself does not have this
    permission. Namespace-scoped Roles limit the grant to test namespaces
    instead of cluster-wide access.
    """
    sa_name = "trustyai-service-operator-controller-manager"
    sa_namespace = "redhat-ods-applications"
    role_name = "trustyai-pods-log-grant"

    with ExitStack() as stack:
        for ns in [evalhub_kueue_model_namespace.name, evalhub_kueue_namespace.name]:
            stack.enter_context(
                cm=Role(
                    client=admin_client,
                    name=role_name,
                    namespace=ns,
                    rules=[{"apiGroups": [""], "resources": ["pods/log"], "verbs": ["get"]}],
                )
            )
            stack.enter_context(
                cm=RoleBinding(
                    client=admin_client,
                    name=f"{role_name}-binding",
                    namespace=ns,
                    subjects_kind="ServiceAccount",
                    subjects_name=sa_name,
                    subjects_namespace=sa_namespace,
                    role_ref_kind="Role",
                    role_ref_name=role_name,
                )
            )
        yield


@pytest.fixture(scope="session")
def evalhub_kueue_model_namespace(
    admin_client: DynamicClient,
    clean_stale_kueue_state: None,
) -> Generator[Namespace, Any, Any]:
    """Namespace for the EvalHub CR and deployment.

    Must NOT carry the evalhub tenant label — TrustyAI rejects EvalHub CRs
    placed in a namespace marked as a tenant namespace.
    """
    with create_ns(
        admin_client=admin_client,
        name=KUEUE_MODEL_NS,
    ) as namespace:
        yield namespace


@pytest.fixture(scope="session")
def evalhub_kueue_cr(
    admin_client: DynamicClient,
    evalhub_kueue_model_namespace: Namespace,
    kueue_unmanaged_dsc: None,
    trustyai_pods_log_rbac: None,
) -> Generator[EvalHub, Any, Any]:
    """Create an EvalHub CR for Kueue tests.

    Depends on kueue_unmanaged_dsc to ensure DSC reconciliation is fully
    complete before creating the EvalHub CR. DSC reconciliation can trigger
    a TrustyAI operator restart; creating the CR during that window causes
    TrustyAI to miss the creation event and never deploy the service.
    """
    if not is_evalhub_crd_available(admin_client):
        pytest.fail(
            "EvalHub CRD 'evalhubs.trustyai.opendatahub.io' not available on this cluster. "
            "Install the TrustyAI/EvalHub operator first."
        )

    with EvalHub(
        client=admin_client,
        name="evalhub-mt",
        namespace=evalhub_kueue_model_namespace.name,
        database={"type": "sqlite"},
        collections=["leaderboard-v2"],
        wait_for_resource=True,
    ) as evalhub:
        yield evalhub


@pytest.fixture(scope="session")
def evalhub_kueue_deployment(
    admin_client: DynamicClient,
    evalhub_kueue_model_namespace: Namespace,
    evalhub_kueue_cr: EvalHub,
) -> Deployment:
    """Wait for the EvalHub deployment to become available."""
    deployment = Deployment(
        client=admin_client,
        name=evalhub_kueue_cr.name,
        namespace=evalhub_kueue_model_namespace.name,
    )
    deployment.wait_for_replicas(timeout=Timeout.TIMEOUT_10MIN)
    return deployment


@pytest.fixture(scope="session")
def evalhub_kueue_route(
    admin_client: DynamicClient,
    evalhub_kueue_model_namespace: Namespace,
    evalhub_kueue_deployment: Deployment,
) -> Route:
    """Get the Route for the EvalHub service."""
    return Route(
        client=admin_client,
        name=evalhub_kueue_deployment.name,
        namespace=evalhub_kueue_model_namespace.name,
        ensure_exists=True,
    )


@pytest.fixture(scope="session")
def evalhub_kueue_ca_bundle_file(
    admin_client: DynamicClient,
) -> str:
    """CA bundle file for verifying TLS on the EvalHub route."""
    return create_ca_bundle_file(client=admin_client)


@pytest.fixture(scope="session")
def installed_cert_manager_operator(admin_client: DynamicClient) -> Generator[None, Any, Any]:
    """Install the cert-manager operator if not already present; uninstall only if this fixture installed it."""
    operator_namespace = "cert-manager-operator"
    package_name = "openshift-cert-manager-operator"
    channel = "stable-v1"

    cert_manager_subscription = Subscription(
        client=admin_client,
        namespace=operator_namespace,
        name=package_name,
    )
    installed_by_fixture = not cert_manager_subscription.exists
    if installed_by_fixture:
        LOGGER.warning(
            "cert-manager not found on this cluster; installing it for this session. "
            "On CI clusters cert-manager is expected to be pre-provisioned — if this "
            "fires in a Jenkins run, the cluster is misconfigured."
        )
        install_operator(
            admin_client=admin_client,
            target_namespaces=None,
            name=package_name,
            channel=channel,
            source="redhat-operators",
            operator_namespace=operator_namespace,
            timeout=Timeout.TIMEOUT_15MIN,
        )
    yield
    if installed_by_fixture:
        uninstall_operator(
            admin_client=admin_client,
            name=package_name,
            operator_namespace=operator_namespace,
            clean_up_namespace=True,
        )


@pytest.fixture(scope="session")
def installed_kueue_operator(
    admin_client: DynamicClient,
    installed_cert_manager_operator: None,
) -> Generator[None, Any, Any]:
    """Install the Red Hat build of Kueue operator if not already present.

    Uninstalls only if this fixture performed the installation.
    """
    package_name = "kueue-operator"
    channel = "stable-v1.3"

    kueue_subscription = Subscription(
        client=admin_client,
        namespace=KUEUE_OPERATOR_NAMESPACE,
        name=package_name,
    )
    installed_by_fixture = not kueue_subscription.exists
    if installed_by_fixture:
        install_operator(
            admin_client=admin_client,
            target_namespaces=None,
            name=package_name,
            channel=channel,
            source="redhat-operators",
            operator_namespace=KUEUE_OPERATOR_NAMESPACE,
            timeout=Timeout.TIMEOUT_15MIN,
        )
    yield
    if installed_by_fixture:
        uninstall_operator(
            admin_client=admin_client,
            name=package_name,
            operator_namespace=KUEUE_OPERATOR_NAMESPACE,
            clean_up_namespace=True,
        )


@pytest.fixture(scope="session")
def kueue_cr(
    admin_client: DynamicClient,
    installed_kueue_operator: None,
) -> Generator[Kueue, Any, Any]:
    """Create the Kueue CR and ensure BatchJob framework integration is enabled.

    If it already exists, adds the BatchJob patch when not already configured
    and waits for the controller rollout to complete before proceeding.
    """
    required_frameworks = {"BatchJob"}
    created_by_fixture = False
    with ExitStack() as stack:
        existing = Kueue(client=admin_client, name="cluster")
        if existing.exists:
            LOGGER.info("Kueue CR 'cluster' already exists, checking frameworks")
            current_frameworks = set(
                (existing.instance.spec.get("config") or {}).get("integrations", {}).get("frameworks", [])
            )
            missing = required_frameworks - current_frameworks
            if missing:
                LOGGER.info(f"Adding missing Kueue frameworks: {missing}")
                updated = list(current_frameworks | required_frameworks)
                baseline_pod_uids = get_kueue_controller_pod_uids(client=admin_client)
                stack.enter_context(
                    cm=ResourceEditor(
                        patches={existing: {"spec": {"config": {"integrations": {"frameworks": updated}}}}}
                    )
                )
                wait_for_kueue_controller_rollout(client=admin_client, baseline_pod_uids=baseline_pod_uids)
            wait_for_kueue_crds_available(client=admin_client)
            yield existing
        else:
            created_by_fixture = True
            kueue = stack.enter_context(
                cm=Kueue(
                    client=admin_client,
                    name="cluster",
                    config={"integrations": {"frameworks": ["BatchJob"]}},
                    management_state="Managed",
                )
            )
            wait_for_kueue_crds_available(client=admin_client)
            yield kueue

        if created_by_fixture:
            full_kueue_controller_cleanup(admin_client=admin_client)


@pytest.fixture(scope="session")
def kueue_unmanaged_dsc(
    dsc_resource: DataScienceCluster,
    kueue_cr: Kueue,
) -> Generator[None, Any, Any]:
    """Ensure RHOAI recognizes the externally installed Kueue operator.

    On a clean RHOAI 3.5 cluster the DSC Kueue component is either Removed or
    Unmanaged. Unmanaged means RHOAI is aware of the external Kueue operator,
    so patch the DSC only when needed; ResourceEditor restores the original
    state at session end.
    """
    try:
        kueue_management_state = dsc_resource.instance.spec.components[DscComponents.KUEUE].managementState
    except (AttributeError, KeyError) as e:
        pytest.fail(f"Kueue component not found in DSC: {e}")

    with ExitStack() as stack:
        if kueue_management_state == DscComponents.ManagementState.UNMANAGED:
            LOGGER.info("DSC Kueue component is already Unmanaged, no patch needed")
        else:
            LOGGER.info(f"Patching DSC Kueue component from {kueue_management_state} to Unmanaged")
            ready_condition = get_dsc_ready_condition(dsc=dsc_resource)
            pre_patch_time = ready_condition.get("lastTransitionTime") if ready_condition else None
            dsc_dict = {
                "spec": {
                    "components": {DscComponents.KUEUE: {"managementState": DscComponents.ManagementState.UNMANAGED}}
                }
            }
            stack.enter_context(cm=ResourceEditor(patches={dsc_resource: dsc_dict}))
            try:
                wait_for_dsc_reconciliation(dsc=dsc_resource, baseline_time=pre_patch_time)
            except TimeoutExpiredError:
                ready_condition = get_dsc_ready_condition(dsc=dsc_resource)
                if not (ready_condition and ready_condition.get("status") == DataScienceCluster.Condition.Status.TRUE):
                    raise
                LOGGER.info("DSC Ready condition never transitioned after the Kueue patch; treating DSC as reconciled")
        yield


@pytest.fixture(scope="session")
def clean_stale_kueue_state(
    admin_client: DynamicClient,
    kueue_unmanaged_dsc: None,
) -> None:
    """Delete test resources left behind by a previous failed run.

    Session fixtures create resources with `with` context managers that fail
    with 409 Conflict if the resources already exist. Namespaces are removed
    first — their Workloads hold the kueue.x-k8s.io/resource-in-use finalizer
    on the ClusterQueues — then the cluster-scoped ClusterQueues and
    ResourceFlavors.
    """
    SLOW_CLEANUP_THRESHOLD = 40

    remove_kueue_visibility_api_services(admin_client=admin_client, wait=True)

    for ns_name in [KUEUE_TENANT_NS, KUEUE_MODEL_NS]:
        namespace = Namespace(client=admin_client, name=ns_name)
        if not namespace.exists:
            continue
        LOGGER.warning(f"Stale namespace {ns_name} found from previous run, cleaning up")
        start = time.monotonic()
        drain_namespace_kueue_resources(admin_client=admin_client, namespace=ns_name)
        namespace.delete(wait=False)
        try:
            for sample in TimeoutSampler(
                wait_timeout=240,
                sleep=5,
                func=lambda n=ns_name: not Namespace(client=admin_client, name=n).exists,
            ):
                elapsed = time.monotonic() - start
                if elapsed > SLOW_CLEANUP_THRESHOLD:
                    LOGGER.warning(
                        f"Stale namespace {ns_name} cleanup is slow ({elapsed:.0f}s elapsed). "
                        f"This usually means Kueue Workloads with finalizers are blocking deletion. "
                        f"If this keeps happening, manually run: "
                        f"oc delete workloads --all -n {ns_name} --force --grace-period=0"
                    )
                if sample:
                    break
        except TimeoutExpiredError:
            LOGGER.warning(f"Namespace {ns_name} stuck after 240s, force-finalizing")
            namespace = Namespace(client=admin_client, name=ns_name)
            if namespace.exists:
                ns_json = namespace.instance.to_dict()
                ns_json["spec"]["finalizers"] = []
                admin_client.request("PUT", f"/api/v1/namespaces/{ns_name}/finalize", body=ns_json)
                try:
                    for done in TimeoutSampler(
                        wait_timeout=30,
                        sleep=2,
                        func=lambda n=ns_name: not Namespace(client=admin_client, name=n).exists,
                    ):
                        if done:
                            break
                except TimeoutExpiredError:
                    LOGGER.warning(
                        f"Namespace {ns_name} still present after force-finalizing; "
                        "continuing, but session fixtures may fail with 409 Conflict"
                    )
        elapsed = time.monotonic() - start
        LOGGER.info(f"Stale namespace {ns_name} cleaned up in {elapsed:.0f}s")

    for resource in [
        ClusterQueue(client=admin_client, name=MULTI_JOB_CLUSTER_QUEUE_NAME),
        ClusterQueue(client=admin_client, name=SINGLE_JOB_CLUSTER_QUEUE_NAME),
        ResourceFlavor(client=admin_client, name=MULTI_JOB_FLAVOR_NAME),
        ResourceFlavor(client=admin_client, name=SINGLE_JOB_FLAVOR_NAME),
    ]:
        if resource.exists:
            LOGGER.warning(f"Stale {resource.kind} {resource.name} found from previous run, cleaning up")
            resource.delete(wait=True)


# Kueue-specific namespace fixture
@pytest.fixture(scope="session")
def evalhub_kueue_namespace(
    admin_client: DynamicClient,
    kueue_unmanaged_dsc: None,
    evalhub_kueue_multi_job_cluster_queue: ClusterQueue,
    evalhub_kueue_single_job_cluster_queue: ClusterQueue,
    clean_stale_kueue_state: None,
) -> Generator[Namespace, Any, Any]:
    """Namespace with both EvalHub tenant label and Kueue opt-in label.

    Depends on both ClusterQueues to enforce teardown ordering: the namespace
    (and all Workloads inside it) must be fully deleted before the ClusterQueues
    are removed. Without this, ClusterQueue deletion fails because Workloads still
    hold the kueue.x-k8s.io/resource-in-use finalizer on the ClusterQueue.
    """
    original_replicas = 0
    try:
        with create_ns(
            admin_client=admin_client,
            name=KUEUE_TENANT_NS,
            labels={EVALHUB_TENANT_LABEL_KEY: "true"},
            add_kueue_label=True,
            delete_timeout=Timeout.TIMEOUT_10MIN,
        ) as namespace:
            yield namespace
            drain_namespace_kueue_resources(admin_client=admin_client, namespace=namespace.name)
            original_replicas = pause_kueue_controller(admin_client=admin_client)
    except ResourceTeardownError:
        LOGGER.warning(
            f"Namespace {KUEUE_TENANT_NS} teardown timed out; clean_stale_kueue_state will recover it on the next run"
        )
    finally:
        resume_kueue_controller(admin_client=admin_client, replicas=original_replicas)


# Multi-job quota fixtures
@pytest.fixture(scope="session")
def evalhub_kueue_multi_job_resource_flavor(
    admin_client: DynamicClient,
    kueue_unmanaged_dsc: None,
    clean_stale_kueue_state: None,
) -> Generator[ResourceFlavor, Any, Any]:
    """ResourceFlavor for multi-job quota tests."""
    with create_resource_flavor(
        name=MULTI_JOB_FLAVOR_NAME,
        client=admin_client,
    ) as resource_flavor:
        yield resource_flavor


@pytest.fixture(scope="session")
def evalhub_kueue_multi_job_cluster_queue(
    admin_client: DynamicClient,
    evalhub_kueue_multi_job_resource_flavor: ResourceFlavor,
    kueue_unmanaged_dsc: None,
) -> Generator[ClusterQueue, Any, Any]:
    """ClusterQueue used by tests that submit more than one EvalHub job.

    Carries the same CPU/memory quota as the single-job queue. It is kept
    separate so that tests which stop or drain one queue do not interfere with
    tests using the other, not because it grants more capacity.
    """
    resource_groups = [
        {
            "coveredResources": ["cpu", "memory"],
            "flavors": [
                {
                    "name": evalhub_kueue_multi_job_resource_flavor.name,
                    "resources": [
                        {"name": "cpu", "nominalQuota": KUEUE_CPU_QUOTA},
                        {"name": "memory", "nominalQuota": KUEUE_MEMORY_QUOTA},
                    ],
                }
            ],
        }
    ]

    with create_cluster_queue(
        name=MULTI_JOB_CLUSTER_QUEUE_NAME,
        client=admin_client,
        resource_groups=resource_groups,
        namespace_selector={},
    ) as cluster_queue:
        wait_for_queue_active(queue=cluster_queue, timeout=Timeout.TIMEOUT_5MIN)
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
        wait_for_queue_active(queue=local_queue)
        yield local_queue


# Single-job quota fixtures (for quota exhaustion tests)
@pytest.fixture(scope="session")
def evalhub_kueue_single_job_resource_flavor(
    admin_client: DynamicClient,
    kueue_unmanaged_dsc: None,
    clean_stale_kueue_state: None,
) -> Generator[ResourceFlavor, Any, Any]:
    """ResourceFlavor for single-job quota tests."""
    with create_resource_flavor(
        name=SINGLE_JOB_FLAVOR_NAME,
        client=admin_client,
    ) as resource_flavor:
        yield resource_flavor


@pytest.fixture(scope="session")
def evalhub_kueue_single_job_cluster_queue(
    admin_client: DynamicClient,
    evalhub_kueue_single_job_resource_flavor: ResourceFlavor,
    kueue_unmanaged_dsc: None,
) -> Generator[ClusterQueue, Any, Any]:
    """ClusterQueue with a small fixed CPU/memory quota for EvalHub Kueue tests.

    The quota does not by itself guarantee that only one job is admitted — Kueue
    admits on the sum of the pods' resource requests, and the EvalHub job payload
    does not set any. Tests that need a job to be gated do so with
    ``stopPolicy: HoldAndDrain`` rather than relying on quota exhaustion. A
    ``pods`` nominalQuota would be the deterministic way to cap concurrency.
    """
    resource_groups = [
        {
            "coveredResources": ["cpu", "memory"],
            "flavors": [
                {
                    "name": evalhub_kueue_single_job_resource_flavor.name,
                    "resources": [
                        {"name": "cpu", "nominalQuota": KUEUE_CPU_QUOTA},
                        {"name": "memory", "nominalQuota": KUEUE_MEMORY_QUOTA},
                    ],
                }
            ],
        }
    ]

    with create_cluster_queue(
        name=SINGLE_JOB_CLUSTER_QUEUE_NAME,
        client=admin_client,
        resource_groups=resource_groups,
        namespace_selector={},
    ) as cluster_queue:
        wait_for_queue_active(queue=cluster_queue, timeout=Timeout.TIMEOUT_5MIN)
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
        wait_for_queue_active(queue=local_queue)
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
        deployment.wait_for_replicas(timeout=Timeout.TIMEOUT_10MIN)
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
        # Bind the ServiceAccount to the ClusterRole that grants this permission.
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


@pytest.fixture(scope="session")
def evalhub_kueue_request_common(
    evalhub_kueue_route: Route,
    evalhub_kueue_user_token: str,
    evalhub_kueue_ca_bundle_file: str,
    evalhub_kueue_namespace: Namespace,
) -> dict[str, str]:
    """Shared EvalHub Kueue request configuration (host, token, CA bundle, tenant)."""
    return {
        "host": evalhub_kueue_route.host,
        "token": evalhub_kueue_user_token,
        "ca_bundle_file": evalhub_kueue_ca_bundle_file,
        "tenant": evalhub_kueue_namespace.name,
    }


@pytest.fixture
def evalhub_job_with_nonexistent_queue(
    evalhub_kueue_namespace: Namespace,
    evalhub_kueue_vllm_service: Service,
    evalhub_kueue_route: Route,
    evalhub_kueue_user_token: str,
    evalhub_kueue_ca_bundle_file: str,
):
    """Fixture that submits a job with non-existent queue and ensures cleanup."""
    payload = build_evalhub_kueue_job_payload(
        queue_name="nonexistent-queue",
        model_service_name=evalhub_kueue_vllm_service.name,
        tenant_namespace=evalhub_kueue_namespace.name,
        job_name="tc-neg-001-invalid-queue",
    )

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
    cleanup_evalhub_job(
        host=evalhub_kueue_route.host,
        token=evalhub_kueue_user_token,
        ca_bundle_file=evalhub_kueue_ca_bundle_file,
        tenant=evalhub_kueue_namespace.name,
        job_id=job_id,
    )


@pytest.fixture
def evalhub_job_without_queue_spec(
    evalhub_kueue_namespace: Namespace,
    evalhub_kueue_vllm_service: Service,
    evalhub_kueue_route: Route,
    evalhub_kueue_user_token: str,
    evalhub_kueue_ca_bundle_file: str,
):
    """Fixture that submits a job without queue spec and ensures cleanup.

    build_evalhub_job_payload deliberately omits the queue field — EvalHub
    must run such jobs as plain batch Jobs without Kueue management.
    """
    payload = build_evalhub_job_payload(
        model_service_name=evalhub_kueue_vllm_service.name,
        tenant_namespace=evalhub_kueue_namespace.name,
        job_name="tc-neg-002-no-queue",
    )

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
    cleanup_evalhub_job(
        host=evalhub_kueue_route.host,
        token=evalhub_kueue_user_token,
        ca_bundle_file=evalhub_kueue_ca_bundle_file,
        tenant=evalhub_kueue_namespace.name,
        job_id=job_id,
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
    payload = build_evalhub_kueue_job_payload(
        queue_name=evalhub_kueue_multi_job_local_queue.name,
        model_service_name=evalhub_kueue_vllm_service.name,
        tenant_namespace=evalhub_kueue_namespace.name,
        job_name="tc-neg-004-cross-tenant",
    )

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
    cleanup_evalhub_job(
        host=evalhub_kueue_route.host,
        token=evalhub_kueue_user_token,
        ca_bundle_file=evalhub_kueue_ca_bundle_file,
        tenant=evalhub_kueue_namespace.name,
        job_id=job_id,
    )
