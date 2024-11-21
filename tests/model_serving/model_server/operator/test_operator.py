import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.exceptions import MissingResourceResError

from utilities.constants import APPLICATIONS_NAMESPACE


@pytest.fixture(scope="class")
def odh_model_controller_deployment(admin_client: DynamicClient):
    deployment = Deployment(
        client=admin_client,
        name="odh-model-controller",
        namespace=APPLICATIONS_NAMESPACE,
    )
    if deployment.exists:
        return deployment

    raise MissingResourceResError(name=deployment.name)


class TestOperator:
    @pytest.mark.smoke
    def test_odh_model_controller_deployment(self, odh_model_controller_deployment):
        # Check odh-model-controller deployment expected number of replicas
        assert odh_model_controller_deployment.instance.spec.replicas == 1

    @pytest.mark.smoke
    def test_odh_model_controller_replicas(self, odh_model_controller_deployment):
        # Check odh-model-controller deployment replicas are running
        odh_model_controller_deployment.wait_for_replicas()
