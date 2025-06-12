import pytest
from ocp_resources.config_map import ConfigMap
from tests.model_explainability.trustyai_operator.utils import validate_trustyai_operator_image


@pytest.mark.smoke
def test_validate_trustyai_operator_image(
    admin_client,
    related_images_refs: set[str],
    trustyai_operator_configmap: ConfigMap,
):
    return validate_trustyai_operator_image(
        client=admin_client,
        related_images_refs=related_images_refs,
        tai_operator_configmap_data=trustyai_operator_configmap.instance.data,
    )
