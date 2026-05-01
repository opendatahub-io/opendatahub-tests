from ocp_resources.deployment import Deployment

from utilities.general import validate_image_format


def validate_trustyai_operator_image(
    related_images_refs: set[str],
    tai_operator_configmap_data: dict[str, str],
    tai_operator_deployment: Deployment,
    upstream: bool = False,
) -> None:
    """Validates the TrustyAI operator image.
    Checks if:
        - container image matches that of the operator configmap.
        - image is present in relatedImages of CSV.
        - image is sourced from the expected registry and pinned w/o tags.

        Args:
            related_images_refs (set[str]): set of related image refs from the CSV
            tai_operator_configmap_data (dict[str, str]): TrustyAI configmap data
            tai_operator_deployment (Deployment): TrustyAI deployment object
            upstream (bool): If True, validate against quay.io/opendatahub instead of registry.redhat.io

        Returns:
            None

        Raises:
            AssertionError: If any of the related images references are not present or invalid.
    """
    tai_operator_image = tai_operator_deployment.instance.spec.template.spec.containers[0].image
    assert tai_operator_image == tai_operator_configmap_data["trustyaiOperatorImage"]
    assert tai_operator_image in related_images_refs
    image_valid, error_message = validate_image_format(image=tai_operator_image, upstream=upstream)
    assert image_valid, error_message
