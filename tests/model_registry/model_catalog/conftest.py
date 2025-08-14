import pytest
from kubernetes.dynamic import DynamicClient
from pytest_testconfig import config as py_config

from ocp_resources.config_map import ConfigMap


@pytest.fixture(scope="function")
def catalog_config_map(request: pytest.FixtureRequest, admin_client: DynamicClient) -> ConfigMap:
    return ConfigMap(name=request.param, client=admin_client, namespace=py_config["applications_namespace"])
