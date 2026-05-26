import pytest
from kubernetes.dynamic import DynamicClient
from pytest_testconfig import config as py_config
from ocp_resources.deployment import Deployment

from tests.model_explainability.trustyai_service.trustyai_service_utils import TRUSTYAI_SERVICE_NAME


@pytest.fixture(scope="class")
def trustyai_operator_deployment(admin_client: DynamicClient) -> Deployment:
    return Deployment(
        client=admin_client,
        name=f"{TRUSTYAI_SERVICE_NAME}-operator-controller-manager",
        namespace=py_config["applications_namespace"],
        ensure_exists=True,
    )
