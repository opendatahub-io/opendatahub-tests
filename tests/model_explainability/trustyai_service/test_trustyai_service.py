from typing import Set

import pytest
from ocp_resources.config_map import ConfigMap
from ocp_resources.namespace import Namespace
from ocp_resources.trustyai_service import TrustyAIService

from tests.model_explainability.trustyai_service.utils import (
    validate_trustyai_service_db_conn_failure,
    validate_trustyai_operator_image,
    validate_trustyai_service_images,
)


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-trustyai-service-invalid-db-cert"},
        )
    ],
    indirect=True,
)
def test_trustyai_service_with_invalid_db_cert(
    admin_client,
    current_client_token,
    model_namespace: Namespace,
    trustyai_service_with_invalid_db_cert,
):
    """Test to make sure TrustyAIService pod fails when incorrect database TLS certificate is used."""
    validate_trustyai_service_db_conn_failure(
        client=admin_client,
        namespace=model_namespace,
        label_selector=f"app.kubernetes.io/instance={trustyai_service_with_invalid_db_cert.name}",
    )


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-validate-trustyai-images"},
        )
    ],
    indirect=True,
)
@pytest.mark.smoke
class TestValidateTrustyAIImages:
    """Test to validate if operator and service image for TrustyAIService matches related images in the CSV.
    Also validate if the image is pinned using a digest and is sourced from registry.redhat.io.
    """

    def test_validate_trustyai_operator_image(
        self,
        admin_client,
        model_namespace: Namespace,
        related_images_refs: Set[str],
        trustyai_operator_configmap: ConfigMap,
    ):
        return validate_trustyai_operator_image(
            client=admin_client,
            related_images_refs=related_images_refs,
            tai_operator_configmap_data=trustyai_operator_configmap.instance.data,
        )

    def test_validate_trustyai_service_image(
        self,
        admin_client,
        model_namespace: Namespace,
        related_images_refs: Set[str],
        trustyai_service_with_pvc_storage: TrustyAIService,
    ):
        return validate_trustyai_service_images(
            client=admin_client,
            related_images_refs=related_images_refs,
            model_namespace=model_namespace,
            label_selector=f"app.kubernetes.io/instance={trustyai_service_with_pvc_storage.name}",
        )
