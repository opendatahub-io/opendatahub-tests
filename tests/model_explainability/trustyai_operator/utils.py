from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment

from tests.model_explainability.trustyai_service.trustyai_service_utils import TRUSTYAI_SERVICE_NAME
from pytest_testconfig import config as py_config
from utilities.general import validate_image_format


def validate_trustyai_operator_image(
    client: DynamicClient, related_images_refs: set[str], tai_operator_configmap_data: dict[str, str]
) -> None:
    """Validates the TrustyAI operator image.
    Checks if:
        - container image matches that of the operator configmap.
        - image is present in relatedImages of CSV.
        - image complies with OpenShift AI requirements i.e. sourced from registry.redhat.io and pinned w/o tags.
    """
    tai_operator_deployment = Deployment(
        client=client,
        name=f"{TRUSTYAI_SERVICE_NAME}-operator-controller-manager",
        namespace=py_config["applications_namespace"],
        wait_for_resource=True,
    )
    tai_operator_image = tai_operator_deployment.instance.spec.template.spec.containers[0].image
    assert tai_operator_image == tai_operator_configmap_data["trustyaiOperatorImage"]
    assert tai_operator_image in related_images_refs
    image_valid, error_message = validate_image_format(image=tai_operator_image)
    assert image_valid, error_message
