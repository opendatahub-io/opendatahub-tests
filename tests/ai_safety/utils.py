import re
from collections.abc import Generator
from typing import Any

from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod

from utilities.general import SHA256_DIGEST_PATTERN
from utilities.infra import create_ns


def create_shared_models_ns(admin_client: DynamicClient, name: str) -> Generator[Namespace, Any, Any]:
    """Create a session-scoped namespace for shared model servers. No teardown — Jenkins handles cleanup."""
    with create_ns(admin_client=admin_client, name=name, teardown=False) as ns:
        yield ns


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
