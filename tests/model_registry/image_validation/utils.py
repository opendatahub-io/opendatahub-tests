import pytest
import structlog
from ocp_resources.pod import Pod

from utilities.general import validate_container_images

LOGGER = structlog.get_logger(name=__name__)


def validate_images(pods_to_validate: list[Pod], related_images_refs: set[str]):
    validation_errors = []
    for pod in pods_to_validate:
        LOGGER.info(f"Validating {pod.name} in {pod.namespace}")
        validation_errors.extend(
            validate_container_images(
                pod=pod,
                valid_image_refs=related_images_refs,
            )
        )

    if validation_errors:
        pytest.fail("\n".join(validation_errors))
