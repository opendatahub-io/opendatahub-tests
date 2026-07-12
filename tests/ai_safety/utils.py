import re

from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.node import Node
from ocp_resources.pod import Pod

from tests.ai_safety.constants import (
    MINIO_IMAGE,
    MINIO_IMAGE_S390X,
    MINIO_MC_IMAGE,
    MINIO_MC_IMAGE_S390X,
)
from utilities.general import SHA256_DIGEST_PATTERN


def validate_tai_component_images(
    pod: Pod, tai_operator_configmap: ConfigMap, include_init_containers: bool = False
) -> None:
    """Validate pod image against tai configmap images and check image for sha256 digest.

    Args:
        pod: Pod
        tai_operator_configmap: ConfigMap
        include_init_containers: bool

    Returns:
        None

    Raises:
        AssertionError: If validation fails.
    """
    tai_configmap_values = tai_operator_configmap.instance.data.values()
    containers = list(pod.instance.spec.containers)
    if include_init_containers:
        containers.extend(pod.instance.spec.initContainers)
    for container in containers:
        assert re.search(SHA256_DIGEST_PATTERN, container.image), (
            f"{container.name} : {container.image} does not have a valid SHA256 digest."
        )
        assert container.image in tai_configmap_values, (
            f"{container.name} : {container.image} not present in TrustyAI operator configmap."
        )


_CLUSTER_ARCH: str | None = None


def get_cluster_architecture(client: DynamicClient) -> str:
    """Return the cluster architecture."""
    global _CLUSTER_ARCH

    if _CLUSTER_ARCH is not None:
        return _CLUSTER_ARCH

    for node in Node.get(dyn_client=client):
        arch = node.instance.metadata.labels.get("kubernetes.io/arch")
        if arch:
            _CLUSTER_ARCH = arch
            return arch

    raise RuntimeError("Unable to determine cluster architecture")


def get_minio_image(client: DynamicClient) -> str:
    """Return the appropriate MinIO image for the current cluster architecture."""
    arch = get_cluster_architecture(client)

    if arch == "s390x":
        return MINIO_IMAGE_S390X

    return MINIO_IMAGE


def get_minio_mc_image(client: DynamicClient) -> str:
    """Return the appropriate MinIO Client image for the current cluster architecture."""
    arch = get_cluster_architecture(client)

    if arch == "s390x":
        return MINIO_MC_IMAGE_S390X

    return MINIO_MC_IMAGE
