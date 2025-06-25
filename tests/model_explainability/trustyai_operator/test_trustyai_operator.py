import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment

from tests.model_explainability.trustyai_operator.utils import validate_trustyai_operator_image


@pytest.mark.smoke
def test_validate_trustyai_operator_image(
    admin_client: DynamicClient,
    related_images_refs: set[str],
    trustyai_operator_configmap: ConfigMap,
    trustyai_operator_deployment: Deployment,
):
    return validate_trustyai_operator_image(
        related_images_refs=related_images_refs,
        tai_operator_configmap_data=trustyai_operator_configmap.instance.data,
        tai_operator_deployment=trustyai_operator_deployment,
    )
