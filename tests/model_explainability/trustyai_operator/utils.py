from ocp_resources.deployment import Deployment

from utilities.general import validate_image_format


def validate_trustyai_operator_image(
    related_images_refs: set[str], tai_operator_configmap_data: dict[str, str], tai_operator_deployment: Deployment
) -> None:
    """Validates the TrustyAI operator image.
    Checks if:
        - container image matches that of the operator configmap.
        - image is present in relatedImages of CSV.
        - image complies with OpenShift AI requirements i.e. sourced from registry.redhat.io and pinned w/o tags.

        Args:
            related_images_refs (set[str]): set of related image refs from the RHOAI CSV
            tai_operator_configmap_data (dict[str, str]): TrustyAI configmap data
            tai_operator_deployment (Deployment): TrustyAI deployment object

        Returns:
            None

        Raises:
            AssertionError: If any of the related images references are not present or invalid.
    """
    tai_operator_image = tai_operator_deployment.instance.spec.template.spec.containers[0].image
    assert tai_operator_image == tai_operator_configmap_data["trustyaiOperatorImage"]
    assert tai_operator_image in related_images_refs
    image_valid, error_message = validate_image_format(image=tai_operator_image)
    assert image_valid, error_message
