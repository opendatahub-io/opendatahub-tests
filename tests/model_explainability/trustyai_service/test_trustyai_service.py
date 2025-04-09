import pytest
from ocp_resources.pod import Pod
from ocp_resources.namespace import Namespace


@pytest.mark.parametrize(
    "model_namespace, trustyai_service_name",
    [
        pytest.param(
            {"name": "test-trustyai-service"},
            "trustyai-service",
        )
    ],
    indirect=["model_namespace"],
)
def test_trustyai_service_with_invalid_db_cert(
    admin_client,
    current_client_token,
    model_namespace: Namespace,
    trustyai_service_name,
    trustyai_service_with_invalid_db_cert,
):
    if pod := list(
        Pod.get(
            dyn_client=admin_client,
            namespace=model_namespace.name,
            label_selector=f"app.kubernetes.io/instance={trustyai_service_name}",
        )
    ):
        for container_status in pod[0].instance.status.containerStatuses:
            if container_status.name == "trustyai-service":
                if container_status.ready and container_status.started:
                    pytest.fail("TrustyAI service is ready when TLS config is invalid.")
    else:
        pytest.fail(f"No pods found for service {trustyai_service_name}.")
