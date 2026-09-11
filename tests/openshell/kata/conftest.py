from collections.abc import Generator
from os import environ
from time import sleep, time
from typing import Any

import grpc
import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.cluster_role_binding import ClusterRoleBinding
from ocp_resources.daemon_set import DaemonSet
from ocp_resources.resource import Resource
from ocp_resources.subscription import Subscription
from ocp_utilities.operators import install_operator, uninstall_operator
from openshell._proto import openshell_pb2
from openshell.sandbox import SandboxClient, SandboxSession
from timeout_sampler import TimeoutSampler

from tests.openshell.kata.constants import KATA_PATCH_MANIFESTS_DIR
from tests.openshell.kata.utils import KataConfig, apply_patch_daemonset, get_kata_config_readiness
from utilities.general import generate_random_name

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="session")
def installed_osc_operator(
    admin_client: DynamicClient,
    teardown_resources: bool,
) -> Generator[None, Any, Any]:
    """Installs the OpenShift sandboxed containers (OSC) Operator from redhat-operators."""
    operator_name = "sandboxed-containers-operator"
    operator_namespace = "openshift-sandboxed-containers-operator"

    osc_subscription = Subscription(client=admin_client, namespace=operator_namespace, name=operator_name)
    installed_by_fixture = not osc_subscription.exists

    if installed_by_fixture:
        LOGGER.info("Installing operator", name=operator_name)
        install_operator(
            admin_client=admin_client,
            target_namespaces=[operator_namespace],
            name=operator_name,
            channel="stable",
            source="redhat-operators",
            operator_namespace=operator_namespace,
            timeout=600,
            install_plan_approval="Automatic",
        )
    else:
        LOGGER.info("Operator already installed, skipping", name=operator_name)

    yield

    if teardown_resources and installed_by_fixture:
        LOGGER.info("Uninstalling operator", name=operator_name)
        uninstall_operator(
            admin_client=admin_client,
            name=operator_name,
            operator_namespace=operator_namespace,
            clean_up_namespace=True,
        )


@pytest.fixture(scope="session")
def kata_config(
    admin_client: DynamicClient,
    installed_osc_operator: None,
    teardown_resources: bool,
) -> Generator[Resource, Any, Any]:
    """Applies the KataConfig and waits for node rollouts to complete."""
    kata_config_resource = KataConfig(
        client=admin_client,
        name="example-kataconfig",
    )

    if not kata_config_resource.exists:
        LOGGER.info("Creating KataConfig...")
        kata_config_resource.res["spec"] = {"enablePeerPods": False, "logLevel": "info"}
        kata_config_resource.create()

    LOGGER.info("Waiting for KataConfig installation to complete (this triggers node reboots)...")

    for sample in TimeoutSampler(
        wait_timeout=600,
        sleep=20,
        func=lambda: get_kata_config_readiness(kata_config=kata_config_resource),
    ):
        LOGGER.debug("KataConfig readiness=%r", sample)
        if sample == "complete":
            LOGGER.info("KataConfig rollout complete.")
            break
        elif sample == "in_progress":
            LOGGER.debug("KataConfig rollout is in progress...")
        else:
            LOGGER.debug("KataConfig status not yet available, waiting...")

    yield kata_config_resource

    if teardown_resources:
        LOGGER.warning("Deleting KataConfig (this will trigger node reboots again)...")
        kata_config_resource.delete()
        kata_config_resource.wait_deleted()


@pytest.fixture(scope="session")
def kata_install_privileged_scc(
    admin_client: DynamicClient,
    installed_osc_operator: None,
    teardown_resources: bool,
) -> Generator[None, Any, Any]:
    """Privileged SCC binding for the kata-install ServiceAccount.

    On ROSA the OSC operator creates this binding automatically.
    On bare-metal it may be missing, so this fixture creates it if absent.
    """
    operator_namespace = "openshift-sandboxed-containers-operator"
    binding_name = "kata-install-privileged-scc"

    crb = ClusterRoleBinding(
        client=admin_client,
        name=binding_name,
    )
    created_by_fixture = not crb.exists

    if created_by_fixture:
        LOGGER.info("Creating privileged SCC binding for kata-install SA")
        crb = ClusterRoleBinding(
            client=admin_client,
            name=binding_name,
            kind_dict={
                "apiVersion": "rbac.authorization.k8s.io/v1",
                "kind": "ClusterRoleBinding",
                "metadata": {"name": binding_name},
                "roleRef": {
                    "apiGroup": "rbac.authorization.k8s.io",
                    "kind": "ClusterRole",
                    "name": "system:openshift:scc:privileged",
                },
                "subjects": [
                    {
                        "kind": "ServiceAccount",
                        "name": "kata-install",
                        "namespace": operator_namespace,
                    }
                ],
            },
        )
        crb.create()
    else:
        LOGGER.info("Privileged SCC binding for kata-install SA already exists")

    yield

    if teardown_resources and created_by_fixture:
        LOGGER.info("Deleting privileged SCC binding for kata-install SA")
        crb.delete()


@pytest.fixture(scope="session")
def kata_initramfs_patches(
    admin_client: DynamicClient,
    kata_config: Resource,
    kata_install_privileged_scc: None,
    teardown_resources: bool,
) -> Generator[None, Any, Any]:
    """Apply nftables and veth kernel module patches to the Kata initramfs.

    These DaemonSets patch the Kata initrd on each worker node to include
    nf_tables and veth modules required by the OpenShell supervisor's
    network init container. This is a temporary workaround until the
    modules are included in the upstream Kata initramfs.
    """
    nftables_manifest = KATA_PATCH_MANIFESTS_DIR / "kata-nftables-patch-job.yaml"
    veth_manifest = KATA_PATCH_MANIFESTS_DIR / "kata-veth-patch-job.yaml"

    patch_daemonsets: list[tuple[Generator[DaemonSet, Any, Any], DaemonSet]] = []
    try:
        for manifest_path in (nftables_manifest, veth_manifest):
            gen = apply_patch_daemonset(
                admin_client=admin_client, manifest_path=manifest_path, teardown=teardown_resources
            )
            ds = next(gen)
            patch_daemonsets.append((gen, ds))

        yield
    finally:
        for gen, _ds in patch_daemonsets:
            try:
                next(gen)
            except StopIteration:
                pass


@pytest.fixture(scope="class")
def kata_sandbox(
    sandbox_client: SandboxClient,
    kata_initramfs_patches: None,
    teardown_resources: bool,
) -> Generator[SandboxSession, Any, Any]:
    """A minimal OpenShell sandbox configured for Kata."""
    template_kwargs: dict[str, Any] = {
        "runtime_class_name": "kata",
    }

    image = environ.get("OPENSHELL_SANDBOX_OPENCODE_IMAGE")
    if image:
        template_kwargs["image"] = image

    spec = openshell_pb2.SandboxSpec(template=openshell_pb2.SandboxTemplate(**template_kwargs))
    sandbox_name = generate_random_name(prefix="kata-shell")

    LOGGER.info("Creating Kata sandbox", name=sandbox_name)
    sandbox_ref = sandbox_client.create(spec=spec, name=sandbox_name)

    try:
        ready_ref = sandbox_client.wait_ready(sandbox_name=sandbox_ref.name, timeout_seconds=90)

        session = SandboxSession(client=sandbox_client, sandbox=ready_ref)

        # Kata VMs need extra time for the supervisor to connect back to the
        # gateway after the pod reaches Ready. Probe with a simple exec until
        # the sandbox actually accepts commands.
        exec_deadline = time() + 120
        last_err = None
        while time() < exec_deadline:
            try:
                session.exec(["true"], timeout_seconds=10)
                LOGGER.info("Kata sandbox exec probe succeeded", name=sandbox_name)
                break
            except grpc.RpcError as exc:
                last_err = exc
                LOGGER.debug("Kata sandbox not yet accepting exec, retrying...", name=sandbox_name, error=str(exc))
                sleep(5)
        else:
            raise RuntimeError(f"Kata sandbox {sandbox_name} never became exec-ready: {last_err}")

        yield session
    finally:
        if teardown_resources:
            LOGGER.info("Deleting Kata sandbox", name=sandbox_ref.name)
            sandbox_client.delete(sandbox_name=sandbox_ref.name)
            sandbox_client.wait_deleted(sandbox_name=sandbox_ref.name)
