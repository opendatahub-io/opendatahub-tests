from ocp_resources.config_map import ConfigMap
from ocp_resources.pod import Pod

from tests.model_explainability.utils import validate_pod_image_against_tai_configmap_images_and_check_digest


def validate_guardrails_orchestrator_images(
    guardrails_orchestrator_pod: Pod, tai_operator_configmap: ConfigMap
) -> None:
    """Validates pod images against tai configmap images and check digest.

    Args:
        guardrails_orchestrator_pod: Pod
        tai_operator_configmap: ConfigMap

    Returns:
        None

    Raises:
        AssertionError: If validation fails.
    """
    validate_pod_image_against_tai_configmap_images_and_check_digest(
        pod=guardrails_orchestrator_pod, tai_operator_configmap=tai_operator_configmap
    )
