import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.pod import Pod

from tests.data_science_pipelines.image_validation.constants import ISTIO_PROXY_CONTAINER_NAME
from utilities.constants import Timeout
from utilities.general import validate_container_images, wait_for_pods_by_labels

LOGGER = structlog.get_logger(name=__name__)


def pod_for_deployment(
    admin_client: DynamicClient,
    namespace: str,
    deployment_name: str,
    label_selector: str,
) -> Pod:
    """Return the pod for a Deployment once the Deployment and its pod exist."""
    Deployment(client=admin_client, name=deployment_name, namespace=namespace).wait(timeout=Timeout.TIMEOUT_5MIN)
    pods = wait_for_pods_by_labels(
        admin_client=admin_client,
        namespace=namespace,
        label_selector=label_selector,
        expected_num_pods=1,
    )
    return pods[0]


def validate_images(pods_to_validate: list[Pod], related_images_refs: set[str]) -> None:
    """Check pod container and init-container images against the live CSV.

    An injected istio-proxy sidecar is skipped by container name. Every other image,
    including the metadata-envoy proxy, must use registry.redhat.io, a sha256 digest,
    and a reference listed in relatedImages.
    """
    validation_errors = []
    for pod in pods_to_validate:
        LOGGER.info(f"Validating {pod.name} in {pod.namespace}")
        validation_errors.extend(
            validate_container_images(
                pod=pod,
                valid_image_refs=related_images_refs,
                skip_container_names={ISTIO_PROXY_CONTAINER_NAME},
            )
        )

    if validation_errors:
        pytest.fail("\n".join(validation_errors))
