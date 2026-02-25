import re

from ocp_resources.config_map import ConfigMap
from ocp_resources.pod import Pod

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
