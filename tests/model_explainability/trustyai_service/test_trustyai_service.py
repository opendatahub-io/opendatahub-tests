import pytest
from ocp_resources.config_map import ConfigMap
from ocp_resources.namespace import Namespace

from tests.model_explainability.trustyai_service.utils import (
    validate_trustyai_service_db_conn_failure,
    validate_trustyai_operator_and_service_images,
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
            {"name": "test-trustyai-operator-and-service-images"},
        )
    ],
    indirect=True,
)
def test_trustyai_operator_and_service_images(
    admin_client,
    current_client_token,
    model_namespace: Namespace,
    trustyai_operator_configmap: ConfigMap,
    trustyai_service_with_db_storage,
):
    """Test to make sure TrustyAIService and TrustyAI Operator pods use the right images from the configmap."""
    validate_trustyai_operator_and_service_images(
        client=admin_client,
        model_namespace=model_namespace.name,
        configmap_data_dict=trustyai_operator_configmap.instance.data,
    )
