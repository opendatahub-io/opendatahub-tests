import re

from ocp_resources.config_map import ConfigMap
from ocp_resources.pod import Pod

from utilities.general import SHA256_DIGEST_PATTERN


def _image_without_registry(image: str) -> str:
    """Strip the registry prefix from a container image reference.

    Cluster policies (e.g. Kyverno) may rewrite the registry while keeping
    the repository path and digest identical.  Comparing images without the
    registry allows validation to succeed across mirror rewrites.

    Args:
        image: Full image reference (e.g. ``registry.redhat.io/rhoai/img@sha256:abc``).

    Returns:
        Image reference without the leading registry
        (e.g. ``rhoai/img@sha256:abc``).
    """
    parts = image.split("/", 1)
    return parts[1] if len(parts) > 1 else image


def validate_tai_component_images(
    pod: Pod, tai_operator_configmap: ConfigMap, include_init_containers: bool = False
) -> None:
    """Validate pod image against tai configmap images and check image for sha256 digest.

    Comparison strips the registry prefix so that cluster-level image
    registry rewrites (e.g. ``registry.redhat.io`` → ``quay.io``) do not
    cause false negatives.

    Args:
        pod: Pod
        tai_operator_configmap: ConfigMap
        include_init_containers: bool

    Returns:
        None

    Raises:
        AssertionError: If validation fails.
    """
    tai_configmap_images = {_image_without_registry(v) for v in tai_operator_configmap.instance.data.values()}
    containers = list(pod.instance.spec.containers)
    if include_init_containers:
        containers.extend(pod.instance.spec.initContainers)
    for container in containers:
        assert re.search(SHA256_DIGEST_PATTERN, container.image), (
            f"{container.name} : {container.image} does not have a valid SHA256 digest."
        )
        assert _image_without_registry(container.image) in tai_configmap_images, (
            f"{container.name} : {container.image} not present in TrustyAI operator configmap."
        )
