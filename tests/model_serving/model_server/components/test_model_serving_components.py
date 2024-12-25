from typing import Dict

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.exceptions import MissingResourceResError
from pytest_testconfig import config as py_config


COMPONENTS_EXPECTED_REPLICAS: Dict[str, int] = {"odh-model-controller": 1, "modelmesh-controller": 3, "etcd": 1}


@pytest.fixture(scope="class")
def component_deployment(request: FixtureRequest, admin_client: DynamicClient) -> Deployment:
    deployment = Deployment(
        client=admin_client,
        name=request.param["name"],
        namespace=py_config["applications_namespace"],
    )
    if deployment.exists:
        return deployment

    raise MissingResourceResError(name=deployment.name)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "component_deployment",
    [
        pytest.param(
            {"name": "odh-model-controller"},
            marks=pytest.mark.polarion("ODS-1919"),
            id="odh-model-controller",
        ),
        pytest.param(
            {"name": "modelmesh-controller"},
            marks=pytest.mark.polarion("ODS-1919"),
            id="modelmesh-controller",
        ),
        pytest.param(
            {"name": "etcd"},
            marks=pytest.mark.polarion("ODS-1919"),
            id="etcd",
        ),
    ],
    indirect=True,
)
class TestOperator:
    def test_deployment_expected_replicas(self, component_deployment):
        """Check expected number of replicas"""
        if expected_replicas := COMPONENTS_EXPECTED_REPLICAS.get(component_deployment.name):
            assert component_deployment.instance.spec.replicas == expected_replicas

        else:
            raise ValueError(f"{component_deployment.name} is missing from `COMPONENTS_EXPECTED_REPLICAS`")

    def test_deployment_running_replicas(self, component_deployment):
        """Check deployment replicas are running"""
        component_deployment.wait_for_replicas()
