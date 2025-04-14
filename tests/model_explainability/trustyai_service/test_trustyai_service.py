import pytest
from ocp_resources.namespace import Namespace

from tests.model_explainability.trustyai_service.utils import wait_for_trustyai_container_terminal_state


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-trustyai-service"},
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
    """
    Makes sure TrustyAIService pod fails when incorrect database TLS certificate is used.
    """
    if terminate_state := wait_for_trustyai_container_terminal_state(
        client=admin_client,
        namespace=model_namespace.name,
        label_selector=f"app.kubernetes.io/instance={trustyai_service_with_invalid_db_cert.name}",
    ):
        assert "Failed to store certificate" in terminate_state.message
    else:
        pytest.fail(f"TrustyAI Service Pod did not fail for {trustyai_service_with_invalid_db_cert.name}.")
