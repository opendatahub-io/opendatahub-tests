from typing import Any, Generator
import os
from kubernetes.dynamic import DynamicClient
import pytest
from tests.model_registry.rest_api.constants import MODEL_REGISTRY_BASE_URI
from tests.model_registry.rest_api.utils import (
    register_model_rest_api,
    execute_model_registry_patch_command,
    create_ca_bundle_file,
)
from utilities.constants import Protocols
from ocp_resources.deployment import Deployment
from tests.model_registry.utils import (
    get_model_registry_deployment_template_dict,
    create_model_registry_instance,
)
from tests.model_registry.constants import (
    DB_RESOURCES_NAME,
    CA_MOUNT_PATH,
    CA_FILE_PATH,
    CA_CONFIGMAP_NAME,
    SECURE_MR_NAME,
    ISTIO_CONFIG_DICT,
    MODEL_REGISTRY_STANDARD_LABELS,
)
from ocp_resources.resource import ResourceEditor
from ocp_resources.secret import Secret
from ocp_resources.config_map import ConfigMap
from simple_logger.logger import get_logger
from ocp_resources.model_registry import ModelRegistry
from pytest_testconfig import config as py_config
from utilities.exceptions import MissingParameter

LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="class")
def model_registry_rest_url(model_registry_instance_rest_endpoint: str) -> str:
    # address and port need to be split in the client instantiation
    return f"{Protocols.HTTPS}://{model_registry_instance_rest_endpoint}"


@pytest.fixture(scope="class")
def model_registry_rest_headers(current_client_token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {current_client_token}",
        "accept": "application/json",
        "Content-Type": "application/json",
    }


@pytest.fixture(scope="class")
def registered_model_rest_api(
    request: pytest.FixtureRequest, model_registry_rest_url: str, model_registry_rest_headers: dict[str, str]
) -> dict[str, Any]:
    return register_model_rest_api(
        model_registry_rest_url=model_registry_rest_url,
        model_registry_rest_headers=model_registry_rest_headers,
        data_dict=request.param,
    )


@pytest.fixture()
def updated_model_registry_resource(
    request: pytest.FixtureRequest,
    model_registry_rest_url: str,
    model_registry_rest_headers: dict[str, str],
    registered_model_rest_api: dict[str, Any],
) -> dict[str, Any]:
    """
    Generic fixture to update any model registry resource via PATCH request.

    Expects request.param to contain:
        - resource_name: Key to identify the resource in registered_model_rest_api
        - api_name: API endpoint name for the resource type
        - data: JSON data to send in the PATCH request

    Returns:
       Dictionary containing the updated resource data
    """
    resource_name = request.param.get("resource_name")
    api_name = request.param.get("api_name")
    if not (api_name and resource_name):
        raise MissingParameter("resource_name and api_name are required parameters for this fixture.")
    resource_id = registered_model_rest_api[resource_name]["id"]
    assert resource_id, f"Resource id not found: {registered_model_rest_api[resource_name]}"
    return execute_model_registry_patch_command(
        url=f"{model_registry_rest_url}{MODEL_REGISTRY_BASE_URI}{api_name}/{resource_id}",
        headers=model_registry_rest_headers,
        data_json=request.param["data"],
    )


@pytest.fixture(scope="class")
def patch_invalid_ca(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    request: pytest.FixtureRequest,
) -> Generator[str, Any, Any]:
    """
    Patches the odh-trusted-ca-bundle ConfigMap with an invalid CA certificate.
    """
    ca_configmap_name = request.param.get("ca_configmap_name", "odh-trusted-ca-bundle")
    ca_file_name = request.param.get("ca_file_name", "invalid-ca.crt")
    ca_file_path = f"{CA_MOUNT_PATH}/{ca_file_name}"
    ca_data = {ca_file_name: "-----BEGIN CERTIFICATE-----\nINVALIDCERTIFICATE\n-----END CERTIFICATE-----"}
    ca_configmap = ConfigMap(
        client=admin_client,
        name=ca_configmap_name,
        namespace=model_registry_namespace,
        ensure_exists=True,
    )
    patch = {
        "metadata": {
            "name": ca_configmap_name,
            "namespace": model_registry_namespace,
        },
        "data": ca_data,
    }
    with ResourceEditor(patches={ca_configmap: patch}):
        yield ca_file_path


@pytest.fixture(scope="class")
def model_registry_instance_ca(
    model_registry_namespace: str,
    model_registry_mysql_config: dict[str, Any],
) -> Generator[ModelRegistry, Any, Any]:
    """
    Deploys a Model Registry instance with a custom CA certificate.
    """
    with create_model_registry_instance(
        namespace=model_registry_namespace,
        name=SECURE_MR_NAME,
        labels=MODEL_REGISTRY_STANDARD_LABELS,
        grpc={},
        rest={},
        istio=ISTIO_CONFIG_DICT,
        mysql=model_registry_mysql_config,
        wait_for_resource=True,
    ) as mr:
        mr.wait_for_condition(condition="Available", status="True")
        yield mr


@pytest.fixture(scope="class")
def mysql_template_with_ca(model_registry_db_secret: Secret) -> dict[str, Any]:
    """
    Patches the MySQL template with the CA file path and volume mount.
    Args:
        model_registry_db_secret: The secret for the model registry's MySQL database
    Returns:
        dict[str, Any]: The patched MySQL template
    """
    mysql_template = get_model_registry_deployment_template_dict(
        secret_name=model_registry_db_secret.name,
        resource_name=DB_RESOURCES_NAME,
    )
    mysql_template["spec"]["containers"][0]["args"].append(f"--ssl-ca={CA_FILE_PATH}")
    mysql_template["spec"]["containers"][0]["volumeMounts"].append({
        "mountPath": CA_MOUNT_PATH,
        "name": CA_CONFIGMAP_NAME,
        "readOnly": True,
    })
    mysql_template["spec"]["volumes"].append({"name": CA_CONFIGMAP_NAME, "configMap": {"name": CA_CONFIGMAP_NAME}})
    return mysql_template


@pytest.fixture(scope="class")
def deploy_secure_mysql_and_mr(
    model_registry_namespace: str,
    model_registry_db_secret: Secret,
    model_registry_db_deployment: Deployment,
    model_registry_mysql_config: dict[str, Any],
    mysql_template_with_ca: dict[str, Any],
) -> Generator[ModelRegistry, None, None]:
    """
    Deploys MySQL with SSL/TLS and a Model Registry configured to use a secure DB connection.

    Ensures the odh-trusted-ca-bundle ConfigMap is mounted in both MySQL and Model Registry deployments,
    and the --ssl-ca argument is set to the correct path.
    """
    patch = {"spec": {"template": mysql_template_with_ca["spec"]}}

    with ResourceEditor(patches={model_registry_db_deployment: patch}):
        with create_model_registry_instance(
            namespace=model_registry_namespace,
            name=SECURE_MR_NAME,
            labels=MODEL_REGISTRY_STANDARD_LABELS,
            grpc={},
            rest={},
            istio=ISTIO_CONFIG_DICT,
            mysql=model_registry_mysql_config,
            wait_for_resource=True,
        ) as mr:
            mr.wait_for_condition(condition="Available", status="True")
            yield mr


@pytest.fixture()
def local_ca_bundle(request: pytest.FixtureRequest, admin_client: DynamicClient) -> Generator[str, Any, Any]:
    """
    Creates a local CA bundle file by fetching the CA bundle from a ConfigMap and appending the router CA from a Secret.
    """
    namespace = getattr(request, "param", {}).get("namespace", py_config["model_registry_namespace"])
    ca_bundle_path = getattr(request, "param", {}).get("ca_bundle_path", "ca-bundle.crt")
    cert_name = getattr(request, "param", {}).get("cert_name", "ca-bundle.crt")
    create_ca_bundle_file(
        admin_client=admin_client, namespace=namespace, ca_bundle_path=ca_bundle_path, cert_name=cert_name
    )
    yield ca_bundle_path

    os.remove(ca_bundle_path)
