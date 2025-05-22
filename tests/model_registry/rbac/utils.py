from typing import Any, Dict
from utilities.constants import Protocols
import logging
from ocp_resources.namespace import Namespace
from model_registry import ModelRegistry as ModelRegistryClient
from tests.model_registry.utils import get_endpoint_from_mr_service, get_mr_service_by_label
from ocp_resources.model_registry import ModelRegistry
from utilities.infra import get_openshift_token
from kubernetes.dynamic import DynamicClient

LOGGER = logging.getLogger(__name__)


def build_mr_client_args(rest_endpoint: str, token: str, author: str) -> Dict[str, Any]:
    """
    Builds arguments for ModelRegistryClient based on REST endpoint and token.

    Args:
        rest_endpoint: The REST endpoint of the Model Registry instance.
        token: The token for the user.
        author: The author of the request.

    Returns:
        A dictionary of arguments for ModelRegistryClient.

    Note: Uses is_secure=False for testing purposes.
    """
    server, port = rest_endpoint.split(":")
    return {
        "server_address": f"{Protocols.HTTPS}://{server}",
        "port": port,
        "user_token": token,
        "is_secure": False,
        "author": author,
    }


def assert_positive_mr_registry(
    model_registry_instance: ModelRegistry,
    model_registry_namespace: str,
    admin_client: DynamicClient,
) -> None:
    """
    Assert that a user has access to the Model Registry.

    Args:
        model_registry_instance: The Model Registry instance to check access for
        model_registry_namespace: The namespace where Model Registry is deployed
        admin_client: The admin client for accessing the cluster

    Raises:
        AssertionError: If client initialization fails
        Exception: If any other error occurs during the check

    Note:
        This function should be called within the appropriate context (admin or user)
        as it uses the current context to get the token.
    """
    token = get_openshift_token()
    namespace_instance = Namespace(client=admin_client, name=model_registry_namespace)
    svc = get_mr_service_by_label(client=admin_client, ns=namespace_instance, mr_instance=model_registry_instance)
    endpoint = get_endpoint_from_mr_service(svc=svc, protocol=Protocols.REST)
    client_args = build_mr_client_args(rest_endpoint=endpoint, token=token, author="rbac-test-user-granted")
    mr_client = ModelRegistryClient(**client_args)
    assert mr_client is not None, "Client initialization failed after granting permissions"
    LOGGER.info("Client instantiated successfully after granting permissions.")


def get_mr_client_args(
    model_registry_instance: ModelRegistry,
    model_registry_namespace: str,
    admin_client: DynamicClient,
    author: str = "rbac-test",
) -> tuple[str, Dict[str, Any]]:
    """
    Get Model Registry client arguments using the current context.

    Args:
        model_registry_instance: The Model Registry instance to connect to
        model_registry_namespace: The namespace where Model Registry is deployed
        admin_client: The admin client for accessing the cluster
        author: The author name for the client (default: "rbac-test")

    Returns:
        Tuple of (token, client_args) for Model Registry client

    Note:
        This function should be called within the appropriate context (admin or user)
        as it uses the current context to get the token.
    """
    token = get_openshift_token()
    namespace_instance = Namespace(client=admin_client, name=model_registry_namespace)
    svc = get_mr_service_by_label(client=admin_client, ns=namespace_instance, mr_instance=model_registry_instance)
    endpoint = get_endpoint_from_mr_service(svc=svc, protocol=Protocols.REST)
    return token, build_mr_client_args(rest_endpoint=endpoint, token=token, author=author)
