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
)
from tests.model_registry.constants import (
    DB_RESOURCES_NAME,
    CA_MOUNT_PATH,
    CA_FILE_PATH,
    CA_CONFIGMAP_NAME,
    ISTIO_CONFIG_DICT,
    MODEL_REGISTRY_STANDARD_LABELS,
    SECURE_MR_NAME,
)
from ocp_resources.resource import ResourceEditor
from ocp_resources.secret import Secret
from ocp_resources.config_map import ConfigMap
from simple_logger.logger import get_logger
from ocp_resources.model_registry import ModelRegistry
from pytest_testconfig import config as py_config
from utilities.exceptions import MissingParameter
import tempfile
from tests.model_registry.rest_api.utils import generate_ca_and_server_cert
import base64


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
    patch_mysql_deployment_with_ssl_ca: Deployment,
) -> Generator[ModelRegistry, None, None]:
    """
    Deploy a secure MySQL and Model Registry instance.
    Args:
        model_registry_namespace: The namespace of the model registry
        model_registry_db_secret: The secret for the model registry's MySQL database
        model_registry_db_deployment: The deployment for the model registry's MySQL database
        model_registry_mysql_config: The MySQL config dictionary
        mysql_template_with_ca: The MySQL template with the CA file path and volume mount
    """
    with ModelRegistry(
        name=SECURE_MR_NAME,
        namespace=model_registry_namespace,
        label=MODEL_REGISTRY_STANDARD_LABELS,
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
    Args:
        request: The pytest request object
        admin_client: The admin client to get the CA bundle from a ConfigMap and append the router CA from a Secret.
    Returns:
        Generator[str, Any, Any]: A generator that yields the CA bundle path.
    """
    ca_bundle_path = getattr(request, "param", {}).get("cert_name", "ca-bundle.crt")
    create_ca_bundle_file(
        admin_client=admin_client,
        namespace=py_config["model_registry_namespace"],
        ca_bundle_path=ca_bundle_path,
        cert_name=ca_bundle_path,
    )
    yield ca_bundle_path

    os.remove(ca_bundle_path)


@pytest.fixture(scope="class")
def test_ca_configmap(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    mysql_ssl_artifacts_and_secrets: dict[str, Any],
) -> Generator[ConfigMap, None, None]:
    """
    Creates a test-specific ConfigMap for the CA bundle, using the generated CA cert.
    Args:
        admin_client: The admin client to create the ConfigMap
        model_registry_namespace: The namespace of the model registry
        mysql_ssl_artifacts_and_secrets: The artifacts and secrets for the MySQL SSL connection
    Returns:
        Generator[ConfigMap, None, None]: A generator that yields the ConfigMap instance.
    """
    with open(mysql_ssl_artifacts_and_secrets["ca_crt"], "r") as f:
        ca_content = f.read()
    if not ca_content:
        LOGGER.info("CA content is empty")
        raise Exception("CA content is empty")
    cm_name = "mysql-ca-configmap"
    with ConfigMap(
        client=admin_client,
        name=cm_name,
        namespace=model_registry_namespace,
        data={"ca-bundle.crt": ca_content},
    ) as cm:
        yield cm


@pytest.fixture(scope="class")
def patch_mysql_deployment_with_ssl_ca(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    model_registry_namespace: str,
    model_registry_db_deployment: Deployment,
    mysql_ssl_artifacts_and_secrets: dict[str, Any],
    test_ca_configmap: ConfigMap,
) -> Generator[Deployment, Any, Any]:
    """
    Patch the MySQL deployment to use the test CA bundle (mysql-ca-configmap),
    and mount the server cert/key for SSL.
    """
    CA_CONFIGMAP_NAME = request.param.get("ca_configmap_name", "mysql-ca-configmap")
    CA_MOUNT_PATH = request.param.get("ca_mount_path", "/etc/mysql/ssl")
    deployment = Deployment(
        client=admin_client,
        name=model_registry_db_deployment.name,
        namespace=model_registry_namespace,
    )
    deployment.wait_for_condition(condition="Available", status="True")
    original_deployment = deployment.instance.to_dict()
    spec = original_deployment["spec"]["template"]["spec"]
    my_sql_container = next(container for container in spec["containers"] if container["name"] == "mysql")
    assert my_sql_container is not None, "Mysql container not found"
    mysql_args = list(my_sql_container.get("args", []))
    mysql_args.extend([
        f"--ssl-ca={CA_MOUNT_PATH}/ca/ca-bundle.crt",
        f"--ssl-cert={CA_MOUNT_PATH}/server_cert/tls.crt",
        f"--ssl-key={CA_MOUNT_PATH}/server_key/tls.key",
    ])

    volumes_mounts = list(my_sql_container.get("volumeMounts", []))
    volumes_mounts.extend([
        {"name": CA_CONFIGMAP_NAME, "mountPath": f"{CA_MOUNT_PATH}/ca", "readOnly": True},
        {
            "name": "mysql-server-cert",
            "mountPath": f"{CA_MOUNT_PATH}/server_cert",
            "readOnly": True,
        },
        {
            "name": "mysql-server-key",
            "mountPath": f"{CA_MOUNT_PATH}/server_key",
            "readOnly": True,
        },
    ])

    my_sql_container["args"] = mysql_args
    my_sql_container["volumeMounts"] = volumes_mounts
    volumes = list(spec["volumes"])
    volumes.extend([
        {"name": CA_CONFIGMAP_NAME, "configMap": {"name": CA_CONFIGMAP_NAME}},
        {"name": "mysql-server-cert", "secret": {"secretName": "mysql-server-cert"}},  # pragma: allowlist secret
        {"name": "mysql-server-key", "secret": {"secretName": "mysql-server-key"}},  # pragma: allowlist secret
    ])

    patch = {"spec": {"template": {"spec": {"volumes": volumes, "containers": [my_sql_container]}}}}
    with ResourceEditor(patches={deployment: patch}):
        deployment.wait_for_condition(condition="Available", status="True")
        yield deployment


@pytest.fixture(scope="class")
def mysql_ssl_artifacts_and_secrets(
    admin_client: DynamicClient,
    model_registry_namespace: str,
) -> Generator[dict[str, Any], None, None]:
    """
    Generates MySQL SSL artifacts and creates corresponding Kubernetes secrets for use in tests.
    This fixture provides file paths and secret objects for the CA certificate, server certificate, and server key.

    Args:
        admin_client: The Kubernetes dynamic client used to create secrets.
        model_registry_namespace: The namespace in which to create the secrets.

    Yields:
        dict: A dictionary containing file paths and secret objects for the CA, server certificate, and server key.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = generate_ca_and_server_cert(tmp_dir=tmp_dir)

        def create_secret(name: str, file_path: str, key_name: str) -> Secret:
            with open(file_path, "rb") as f:
                file_content_raw_bytes = f.read()
                file_content_b64_string = base64.b64encode(file_content_raw_bytes).decode("utf-8")
                data = {key_name: file_content_b64_string}
            secret = Secret(
                client=admin_client,
                name=name,
                namespace=model_registry_namespace,
                data_dict=data,
                wait_for_resource=True,
            )
            secret.create()
            return secret

        ca_secret = create_secret(name="mysql-ca", file_path=paths["ca_crt"], key_name="ca.crt")
        server_cert_secret = create_secret(name="mysql-server-cert", file_path=paths["server_crt"], key_name="tls.crt")
        server_key_secret = create_secret(name="mysql-server-key", file_path=paths["server_key"], key_name="tls.key")

        yield {
            "ca_crt": paths["ca_crt"],
            "server_crt": paths["server_crt"],
            "server_key": paths["server_key"],
            "ca_secret": ca_secret,
            "server_cert_secret": server_cert_secret,
            "server_key_secret": server_key_secret,
        }
