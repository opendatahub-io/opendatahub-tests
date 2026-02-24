import copy
import json
import os
from typing import Any

import requests
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from pyhelper_utils.shell import run_command
from simple_logger.logger import get_logger

from tests.model_registry.exceptions import (
    ModelRegistryResourceNotCreated,
    ModelRegistryResourceNotUpdated,
)
from tests.model_registry.model_registry.rest_api.constants import MODEL_REGISTER_DATA, MODEL_REGISTRY_BASE_URI
from utilities.exceptions import ResourceValueMismatch

LOGGER = get_logger(name=__name__)


def execute_model_registry_patch_command(
    url: str, headers: dict[str, str], data_json: dict[str, Any]
) -> dict[Any, Any]:
    resp = requests.patch(url=url, json=data_json, headers=headers, verify=False, timeout=60)
    LOGGER.info(f"url: {url}, status code: {resp.status_code}, rep: {resp.text}")

    if resp.status_code != 200:
        raise ModelRegistryResourceNotUpdated(
            f"Failed to update ModelRegistry resource: {url}, {resp.status_code}: {resp.text}"
        )
    try:
        return json.loads(resp.text)
    except json.JSONDecodeError:
        LOGGER.error(f"Unable to parse {resp.text}")
        raise


def execute_model_registry_post_command(
    url: str, headers: dict[str, str], data_json: dict[str, Any], verify: bool | str = False
) -> dict[Any, Any]:
    resp = requests.post(url=url, json=data_json, headers=headers, verify=verify, timeout=60)
    LOGGER.info(f"url: {url}, status code: {resp.status_code}, rep: {resp.text}")

    if resp.status_code not in [200, 201]:
        raise ModelRegistryResourceNotCreated(
            f"Failed to create ModelRegistry resource: {url}, {resp.status_code}: {resp.text}"
        )
    try:
        return json.loads(resp.text)
    except json.JSONDecodeError:
        LOGGER.error(f"Unable to parse {resp.text}")
        raise


def register_model_rest_api(
    model_registry_rest_url: str,
    model_registry_rest_headers: dict[str, str],
    data_dict: dict[str, Any],
    verify: bool | str = False,
) -> dict[str, Any]:
    # register a model
    register_model = execute_model_registry_post_command(
        url=f"{model_registry_rest_url}{MODEL_REGISTRY_BASE_URI}registered_models",
        headers=model_registry_rest_headers,
        data_json=data_dict["register_model_data"],
        verify=verify,
    )
    # create associated model version:
    model_data = data_dict["model_version_data"]
    model_data["registeredModelId"] = register_model["id"]
    model_version = execute_model_registry_post_command(
        url=f"{model_registry_rest_url}{MODEL_REGISTRY_BASE_URI}model_versions",
        headers=model_registry_rest_headers,
        data_json=model_data,
        verify=verify,
    )
    # create associated model artifact
    model_artifact = execute_model_registry_post_command(
        url=f"{model_registry_rest_url}{MODEL_REGISTRY_BASE_URI}model_versions/{model_version['id']}/artifacts",
        headers=model_registry_rest_headers,
        data_json=data_dict["model_artifact_data"],
        verify=verify,
    )
    LOGGER.info(
        f"Successfully registered model: {register_model}, with version: {model_version} and "
        f"associated artifact: {model_artifact}"
    )
    return {"register_model": register_model, "model_version": model_version, "model_artifact": model_artifact}


def validate_resource_attributes(
    expected_params: dict[str, Any], actual_resource_data: dict[str, Any], resource_name: str
) -> None:
    """
    Validate that expected parameters match actual resource data.
    Args:
       expected_params: Dictionary of expected attribute values
       actual_resource_data: Dictionary of actual resource data from API
       resource_name: Name of the resource being validated for error messages

    Raises:
        ResourceValueMismatch: When expected and actual values don't match

    """
    errors: list[dict[str, list[Any]]]
    if errors := [
        {key: [f"Expected value: {expected_params[key]}, actual value: {actual_resource_data.get(key)}"]}
        for key in expected_params
        if (not actual_resource_data.get(key) or actual_resource_data[key] != expected_params[key])
    ]:
        raise ResourceValueMismatch(f"Resource: {resource_name} has mismatched data: {errors}")
    LOGGER.info(f"Successfully validated resource: {resource_name}: {actual_resource_data['name']}")


def generate_ca_and_server_cert(
    tmp_dir: str,
    db_service_hostname: str = "db-model-registry.rhoai-model-registries.svc.cluster.local",
    ca_name: str = "Test CA",
    server_cn: str = "mysql-server",
) -> dict[str, str]:
    """
    Generates a CA and server certificate/key for the MySQL server.

    Args:
        tmp_dir: The temporary directory to store the certificates.
        db_service_hostname: The hostname of the MySQL server.
        ca_name: The name of the CA.
        server_cn: The common name of the server.

    Returns:
        Dict[str, str]: A dictionary containing the paths to the CA certificate, server key, and server certificate.
    """

    ca_key = os.path.join(tmp_dir, "ca.key")
    ca_crt = os.path.join(tmp_dir, "ca.crt")
    server_key = os.path.join(tmp_dir, "server-key.pem")
    server_csr = os.path.join(tmp_dir, "server.csr")
    server_crt = os.path.join(tmp_dir, "server-cert.pem")

    LOGGER.info(f"Generating CA and server cert in {tmp_dir} for DB hostname {db_service_hostname}")

    create_ca_key_and_cert_with_openssl(ca_key=ca_key, ca_crt=ca_crt, ca_name=ca_name)
    generate_db_server_key_and_csr_with_openssl(server_key=server_key, server_csr=server_csr, server_cn=server_cn)
    sign_db_server_cert_with_ca_with_openssl(server_crt=server_crt, server_csr=server_csr, ca_crt=ca_crt, ca_key=ca_key)

    return {
        "ca_crt": ca_crt,
        "server_key": server_key,
        "server_crt": server_crt,
    }


def create_ca_key_and_cert_with_openssl(
    ca_key: str,
    ca_crt: str,
    ca_name: str,
) -> None:
    """
    Creates a CA private key and certificate.

    Args:
        ca_key: The path to the CA private key.
        ca_crt: The path to the CA certificate.
        ca_name: The name of the CA.

    Returns:
        None
    """
    run_command(command=["openssl", "genrsa", "-out", ca_key, "2048"], check=True)
    run_command(
        command=[
            "openssl",
            "req",
            "-x509",
            "-new",
            "-nodes",
            "-key",
            ca_key,
            "-sha256",
            "-days",
            "3650",
            "-out",
            ca_crt,
            "-subj",
            f"/CN={ca_name}",
        ],
        check=True,
    )


def generate_db_server_key_and_csr_with_openssl(
    server_key: str,
    server_csr: str,
    server_cn: str,
) -> None:
    """
    Generates a DB server private key and CSR.

    Args:
        server_key: The path to the DB server private key.
        server_csr: The path to the DB server CSR.
        server_cn: The common name of the DB server.

    Returns:
        None
    """
    run_command(command=["openssl", "genrsa", "-out", server_key, "2048"], check=True)
    run_command(
        command=["openssl", "req", "-new", "-key", server_key, "-out", server_csr, "-subj", f"/CN={server_cn}"],
        check=True,
    )


def sign_db_server_cert_with_ca_with_openssl(
    server_crt: str,
    server_csr: str,
    ca_crt: str,
    ca_key: str,
) -> None:
    """
    Signs a DB server certificate with a CA.

    Args:
        server_crt: The path to the DB server certificate.
        server_csr: The path to the DB server CSR.
        ca_crt: The path to the CA certificate.
        ca_key: The path to the CA private key.

    Returns:
        None
    """
    run_command(
        command=[
            "openssl",
            "x509",
            "-req",
            "-in",
            server_csr,
            "-CA",
            ca_crt,
            "-CAkey",
            ca_key,
            "-CAcreateserial",
            "-out",
            server_crt,
            "-days",
            "3650",
            "-sha256",
        ],
        check=True,
    )


def get_register_model_data(num_models: int) -> list[dict[str, Any]]:
    model_data = []
    for num_model in range(num_models):
        copy_data = copy.deepcopy(MODEL_REGISTER_DATA)
        for value in copy_data.values():
            value["name"] = f"{value['name']}{num_model}"
            value["description"] = f"{value['description']}{num_model}"
        model_data.append(copy_data)
    return model_data


def get_mr_deployment(admin_client: DynamicClient, mr_namespace: str) -> list[Deployment]:
    return list(Deployment.get(client=admin_client, namespace=mr_namespace))
