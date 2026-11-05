import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from pytest_testconfig import config as py_config

from utilities.certificates_utils import create_ca_bundle_file
from utilities.constants import TRUSTYAI_SERVICE_NAME


@pytest.fixture(scope="session")
def trustyai_operator_configmap(
    admin_client: DynamicClient,
) -> ConfigMap:
    return ConfigMap(
        client=admin_client,
        namespace=py_config["applications_namespace"],
        name=f"{TRUSTYAI_SERVICE_NAME}-operator-config",
        ensure_exists=True,
    )


@pytest.fixture(scope="class")
def openshift_ca_bundle_file(
    admin_client: DynamicClient,
) -> str:
    """Create CA bundle file for HTTPS verification."""
    return create_ca_bundle_file(client=admin_client)
