import shlex
import subprocess
from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from model_registry import ModelRegistry as ModelRegistryClient
from model_registry.types import RegisteredModel
from ocp_resources.deployment import Deployment
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.service_account import ServiceAccount
from pyhelper_utils.shell import run_command
from pytest import FixtureRequest
from simple_logger.logger import get_logger

from tests.model_registry.constants import (
    MODEL_REGISTRY_POD_FILTER,
    MR_INSTANCE_NAME,
)
from tests.model_registry.utils import (
    get_endpoint_from_mr_service,
    get_mr_service_by_label,
)
from utilities.constants import Protocols
from utilities.general import wait_for_pods_by_labels

LOGGER = get_logger(name=__name__)
DEFAULT_TOKEN_DURATION = "10m"


@pytest.fixture(scope="class")
def model_registry_instance_rest_endpoint(admin_client: DynamicClient, model_registry_instance) -> list[str]:
    """
    Get the REST endpoint(s) for the model registry instance.
    """
    # get all the services:
    mr_services = [
        get_mr_service_by_label(client=admin_client, namespace_name=mr_instance.namespace, mr_instance=mr_instance)
        for mr_instance in model_registry_instance
    ]
    if not mr_services:
        raise ResourceNotFoundError("No model registry services found")
    return [get_endpoint_from_mr_service(svc=svc, protocol=Protocols.REST) for svc in mr_services]


@pytest.fixture(scope="class")
def model_registry_client(
    current_client_token: str,
    model_registry_instance_rest_endpoint: list[str],
) -> list[ModelRegistryClient]:
    """
    Get a client for the model registry instance.
    Args:
        current_client_token: The current client token
        model_registry_instance_rest_endpoint: list of model registry endpoints
    Returns:
        ModelRegistryClient: A client for the model registry instance
    """
    mr_clients = []
    for rest_endpoint in model_registry_instance_rest_endpoint:
        server, port = rest_endpoint.split(":")
        mr_clients.append(
            ModelRegistryClient(
                server_address=f"{Protocols.HTTPS}://{server}",
                port=int(port),
                author="opendatahub-test",
                user_token=current_client_token,
                is_secure=False,
            )
        )
    if not mr_clients:
        raise ResourceNotFoundError("No model registry clients created")
    return mr_clients


@pytest.fixture(scope="class")
def registered_model(
    request: FixtureRequest, model_registry_client: list[ModelRegistryClient]
) -> Generator[RegisteredModel]:
    yield model_registry_client[0].register_model(
        name=request.param.get("model_name"),
        uri=request.param.get("model_uri"),
        version=request.param.get("model_version"),
        version_description=request.param.get("model_description"),
        model_format_name=request.param.get("model_format"),
        model_format_version=request.param.get("model_format_version"),
        storage_key=request.param.get("model_storage_key"),
        storage_path=request.param.get("model_storage_path"),
        metadata=request.param.get("model_metadata"),
    )


@pytest.fixture(scope="class")
def model_registry_rest_url(model_registry_instance_rest_endpoint: list[str]) -> list[str]:
    # address and port need to be split in the client instantiation
    return [f"{Protocols.HTTPS}://{rest_url}" for rest_url in model_registry_instance_rest_endpoint]


@pytest.fixture(scope="class")
def model_registry_deployment_containers(model_registry_namespace: str) -> list[dict[str, Any]]:
    return Deployment(
        name=MR_INSTANCE_NAME, namespace=model_registry_namespace, ensure_exists=True
    ).instance.spec.template.spec.containers


@pytest.fixture(scope="class")
def model_registry_pod(admin_client: DynamicClient, model_registry_namespace: str) -> Pod:
    return wait_for_pods_by_labels(
        admin_client=admin_client,
        namespace=model_registry_namespace,
        label_selector=MODEL_REGISTRY_POD_FILTER,
        expected_num_pods=1,
    )[0]


@pytest.fixture(scope="class")
def sa_token(service_account: ServiceAccount) -> str:
    """
    Retrieves a short-lived token for the ServiceAccount using 'oc create token'.
    """
    sa_name = service_account.name
    namespace = service_account.namespace
    LOGGER.info(f"Retrieving token for ServiceAccount: {sa_name} in namespace {namespace}")
    try:
        cmd = f"oc create token {sa_name} -n {namespace} --duration={DEFAULT_TOKEN_DURATION}"
        LOGGER.debug(f"Executing command: {cmd}")
        _, out, _ = run_command(command=shlex.split(cmd), verify_stderr=False, check=True, timeout=30)
        token = out.strip()
        if not token:
            raise ValueError("Retrieved token is empty after successful command execution.")

        LOGGER.info(f"Successfully retrieved token for SA '{sa_name}'")
        return token  # noqa: TRY300

    except Exception as e:  # Catch all exceptions from the try block
        error_type = type(e).__name__
        log_message = (
            f"Failed during token retrieval for SA '{sa_name}' in namespace '{namespace}'. "
            f"Error Type: {error_type}, Message: {e!s}"
        )
        if isinstance(e, subprocess.CalledProcessError):
            # Add specific details for CalledProcessError
            # run_command already logs the error if log_errors=True and returncode !=0,
            # but we can add context here.
            stderr_from_exception = e.stderr.strip() if e.stderr else "N/A"
            log_message += f". Exit Code: {e.returncode}. Stderr from exception: {stderr_from_exception}"
        elif isinstance(e, subprocess.TimeoutExpired):
            timeout_value = getattr(e, "timeout", "N/A")
            log_message += f". Command timed out after {timeout_value} seconds."
        elif isinstance(e, FileNotFoundError):
            # This occurs if 'oc' is not found.
            # e.filename usually holds the name of the file that was not found.
            command_not_found = e.filename if hasattr(e, "filename") and e.filename else shlex.split(cmd)[0]
            log_message += f". Command '{command_not_found}' not found. Is it installed and in PATH?"

        LOGGER.error(log_message)
        raise


@pytest.fixture(scope="class")
def mr_access_role(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    sa_namespace: Namespace,
) -> Generator[Role]:
    """
    Creates the MR Access Role using direct constructor parameters and a context manager.
    """
    role_name = f"registry-user-{MR_INSTANCE_NAME}-{sa_namespace.name[:8]}"
    LOGGER.info(f"Defining Role: {role_name} in namespace {model_registry_namespace}")

    role_rules: list[dict[str, Any]] = [
        {
            "apiGroups": [""],
            "resources": ["services"],
            "resourceNames": [MR_INSTANCE_NAME],  # Grant access only to the specific MR service object
            "verbs": ["get"],
        }
    ]
    role_labels = {
        "app.kubernetes.io/component": "model-registry-test-rbac",
        "test.opendatahub.io/namespace": sa_namespace.name,
    }

    with Role(
        client=admin_client,
        name=role_name,
        namespace=model_registry_namespace,
        rules=role_rules,
        label=role_labels,
        wait_for_resource=True,
    ) as role:
        yield role


@pytest.fixture(scope="class")
def mr_access_role_binding(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    mr_access_role: Role,
    sa_namespace: Namespace,
) -> Generator[RoleBinding]:
    """
    Creates the MR Access RoleBinding using direct constructor parameters and a context manager.
    """
    binding_name = f"{mr_access_role.name}-binding"

    LOGGER.info(
        f"Defining RoleBinding: {binding_name} linking Group 'system:serviceaccounts:{sa_namespace.name}' "
        f"to Role '{mr_access_role.name}' in namespace {model_registry_namespace}"
    )
    binding_labels = {
        "app.kubernetes.io/component": "model-registry-test-rbac",
        "test.opendatahub.io/namespace": sa_namespace.name,
    }

    with RoleBinding(
        client=admin_client,
        name=binding_name,
        namespace=model_registry_namespace,
        # Subject parameters
        subjects_kind="Group",
        subjects_name=f"system:serviceaccounts:{sa_namespace.name}",
        subjects_api_group="rbac.authorization.k8s.io",  # This is the default apiGroup for Group kind
        # Role reference parameters
        role_ref_kind=mr_access_role.kind,
        role_ref_name=mr_access_role.name,
        label=binding_labels,
        wait_for_resource=True,
    ) as binding:
        LOGGER.info(f"RoleBinding {binding.name} created successfully.")
        yield binding
        LOGGER.info(f"RoleBinding {binding.name} deletion initiated by context manager.")
