import pytest
from kubernetes.dynamic import DynamicClient
from pytest_testconfig import config as py_config
from ocp_resources.dsc_initialization import DSCInitialization


@pytest.fixture(scope="session")
def dsci_resource(admin_client: DynamicClient) -> DSCInitialization:
    return DSCInitialization(client=admin_client, name=py_config["dsci_name"], ensure_exists=True)
