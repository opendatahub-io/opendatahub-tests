from collections.abc import Generator
from pathlib import Path
from typing import Any

import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.daemon_set import DaemonSet
from ocp_resources.pod import Pod
from ocp_resources.resource import Resource
from timeout_sampler import TimeoutSampler
from yaml import safe_load

from tests.openshell.image_constants import OpenShellImages

LOGGER = structlog.get_logger(name=__name__)


class KataConfig(Resource):
    api_group: str = "kataconfiguration.openshift.io"


class RuntimeClass(Resource):
    api_group: str = "node.k8s.io"


def get_kata_config_readiness(kata_config: Resource) -> str | None:
    """Check KataConfig readiness using the documented status conditions.

    Returns:
        "complete" if InProgress condition is False and all kataNodes are installed,
        "in_progress" if installation is still running,
        or None if status is not yet populated.
    """
    status = kata_config.instance.get("status", {})

    conditions = status.get("conditions", [])
    in_progress = next(
        (c for c in conditions if c.get("type") == "InProgress"),
        None,
    )

    if in_progress is None:
        return None

    if in_progress.get("status") == "True":
        return "in_progress"

    kata_nodes = status.get("kataNodes", {})
    installed_nodes = kata_nodes.get("installed", [])
    installing_nodes = kata_nodes.get("installing", [])
    waiting_nodes = kata_nodes.get("waitingToInstall", [])
    if installing_nodes or waiting_nodes:
        return "in_progress"

    node_count = kata_nodes.get("nodeCount", 0)
    if node_count > 0 and len(installed_nodes) < node_count:
        return "in_progress"

    return "complete"


def apply_patch_daemonset(
    admin_client: DynamicClient,
    manifest_path: Path,
    teardown: bool,
) -> Generator[DaemonSet, Any, Any]:
    """Apply a Kata initramfs patch DaemonSet, wait for all pods to finish, then clean up."""
    with open(manifest_path) as manifest_file:
        manifest = safe_load(stream=manifest_file)

    manifest["spec"]["template"]["spec"]["containers"][0]["image"] = OpenShellImages.UBI9_MINIMAL

    ds_name = manifest["metadata"]["name"]
    ds_namespace = manifest["metadata"]["namespace"]

    existing_ds = DaemonSet(
        client=admin_client,
        name=ds_name,
        namespace=ds_namespace,
    )
    if existing_ds.exists:
        LOGGER.info("Patch DaemonSet already exists, deleting stale instance", name=ds_name)
        existing_ds.delete()
        existing_ds.wait_deleted()

    LOGGER.info("Applying patch DaemonSet", manifest=manifest_path.name)
    with DaemonSet(
        client=admin_client,
        name=ds_name,
        namespace=ds_namespace,
        kind_dict=manifest,
        teardown=teardown,
    ) as ds:
        LOGGER.info("Waiting for patch pods to complete", name=ds_name)
        for sample in TimeoutSampler(
            wait_timeout=180,
            sleep=15,
            func=lambda: _all_patch_pods_done(admin_client=admin_client, ds=ds),
        ):
            if sample:
                LOGGER.info("All patch pods have completed", name=ds_name)
                break

        yield ds


def _all_patch_pods_done(admin_client: DynamicClient, ds: DaemonSet) -> bool:
    """Check that the DaemonSet has scheduled all pods and all are Running."""
    status = ds.instance.get("status", {})
    desired = status.get("desiredNumberScheduled", 0)
    ready = status.get("numberReady", 0)
    if desired == 0:
        return False
    if ready < desired:
        LOGGER.debug("Patch DaemonSet pods ready", name=ds.name, ready=ready, desired=desired)
        return False

    pods = list(
        Pod.get(
            dyn_client=admin_client,
            namespace=ds.namespace,
            label_selector=f"app={ds.name}",
        )
    )
    for pod in pods:
        logs = pod.log()
        if "Patch complete. Sleeping." in logs or "already present in initramfs, skipping" in logs:
            continue
        LOGGER.debug("Pod not yet done", name=pod.name)
        return False

    return True
