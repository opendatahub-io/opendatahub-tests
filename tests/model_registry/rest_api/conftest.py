from typing import Any, Generator

import pytest

from tests.model_registry.rest_api.constants import MODEL_REGISTRY_BASE_URI
from tests.model_registry.rest_api.utils import register_model_rest_api, execute_model_registry_patch_command
from utilities.constants import Protocols
from ocp_resources.deployment import Deployment
from ocp_resources.service import Service
from tests.model_registry.utils import get_model_registry_deployment_template_dict, create_secure_model_registry

from tests.model_registry.constants import (
    DB_RESOURCES_NAME,
    CA_CONFIGMAP_NAME,
    CA_MOUNT_PATH,
    CA_FILE_PATH,
)
from ocp_resources.resource import ResourceEditor
from ocp_resources.secret import Secret


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


@pytest.fixture(scope="class")
def updated_model_artifact(
    request: pytest.FixtureRequest,
    model_registry_rest_url: str,
    model_registry_rest_headers: dict[str, str],
    registered_model_rest_api: dict[str, Any],
) -> dict[str, Any]:
    model_artifact_id = registered_model_rest_api["model_artifact"]["id"]
    assert model_artifact_id, f"Model artifact id not found: {registered_model_rest_api['model_artifact']}"
    return execute_model_registry_patch_command(
        url=f"{model_registry_rest_url}{MODEL_REGISTRY_BASE_URI}model_artifacts/{model_artifact_id}",
        headers=model_registry_rest_headers,
        data_json=request.param,
    )


@pytest.fixture(scope="class")
def deploy_secure_mysql_and_mr(
    model_registry_namespace: str,
    model_registry_db_secret: Secret,
    model_registry_db_service: Service,
    model_registry_db_deployment: Deployment,
) -> Generator[None, None, None]:
    """
    Fixture to deploy MySQL with SSL/TLS and a Model Registry configured to use a secure DB connection.

    Parameters:
        admin_client (object): Kubernetes/OpenShift admin client for resource operations.
        model_registry_namespace (str): Namespace where resources are deployed.
        model_registry_db_secret (Secret): Secret resource for MySQL credentials.
        model_registry_db_service (Service): Service resource for MySQL DB.
        model_registry_db_deployment (Deployment): Deployment resource for MySQL DB.
    """

    mysql_template = get_model_registry_deployment_template_dict(
        secret_name=model_registry_db_secret.name,
        resource_name=DB_RESOURCES_NAME,
    )

    ssl_ca_arg = f"--ssl-ca={CA_FILE_PATH}"
    ca_volume_mount = {"mountPath": CA_MOUNT_PATH, "name": CA_CONFIGMAP_NAME, "readOnly": True}
    ca_volume = {"name": CA_CONFIGMAP_NAME, "configMap": {"name": CA_CONFIGMAP_NAME}}

    mysql_template["spec"]["containers"][0]["args"] += [ssl_ca_arg]
    mysql_template["spec"]["containers"][0]["volumeMounts"].append(ca_volume_mount)
    mysql_template["spec"]["volumes"].append(ca_volume)

    patch = {"spec": {"template": mysql_template["spec"]}}

    with ResourceEditor(patches={model_registry_db_deployment: patch}):
        with create_secure_model_registry(
            model_registry_namespace=model_registry_namespace,
            model_registry_db_service=model_registry_db_service,
            model_registry_db_secret=model_registry_db_secret,
            ca_file_path=CA_FILE_PATH,
        ) as mr:
            mr.wait_for_condition(condition="Available", status="True")
            yield mr
