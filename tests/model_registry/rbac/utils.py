from typing import Any, Dict, Generator

from kubernetes.dynamic import DynamicClient
from timeout_sampler import TimeoutSampler

from ocp_resources.deployment import Deployment
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from utilities.constants import Protocols
import logging
from model_registry import ModelRegistry as ModelRegistryClient
from utilities.infra import get_openshift_token

LOGGER = logging.getLogger(__name__)


def build_mr_client_args(rest_endpoint: str, token: str, author: str = "rbac-test") -> Dict[str, Any]:
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
    model_registry_instance_rest_endpoint: str,
    token: str = "",
) -> None:
    """
    Assert that a user has access to the Model Registry.

    Args:
        model_registry_instance_rest_endpoint: The Model Registry rest endpoint
        token: user token

    Raises:
        AssertionError: If client initialization fails
        Exception: If any other error occurs during the check

    Note:
        This function should be called within the appropriate context (admin or user)
        as it uses the current context to get the token.
    """
    client_args = build_mr_client_args(
        rest_endpoint=model_registry_instance_rest_endpoint,
        token=token or get_openshift_token(),
        author="rbac-test-user-granted",
    )
    mr_client = ModelRegistryClient(**client_args)
    assert mr_client is not None, "Client initialization failed after granting permissions"
    LOGGER.info("Client instantiated successfully after granting permissions.")


def wait_for_oauth_openshift_deployment() -> None:
    deployment_obj = Deployment(name="oauth-openshift", namespace="openshift-authentication", ensure_exists=True)

    _log = f"Wait for {deployment_obj.name} -> Type: Progressing -> Reason:"

    def _wait_sampler(_reason: str) -> None:
        sampler = TimeoutSampler(
            wait_timeout=240,
            sleep=5,
            func=lambda: deployment_obj.instance.status.conditions,
        )
        for sample in sampler:
            for _spl in sample:
                if _spl.type == "Progressing" and _spl.reason == _reason:
                    return

    for reason in ("ReplicaSetUpdated", "NewReplicaSetAvailable"):
        LOGGER.info(f"{_log} {reason}")
        _wait_sampler(_reason=reason)


def create_role_binding(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    mr_access_role: Role,
    name: str,
    subjects_kind: str,
    subjects_name: str,
) -> Generator[RoleBinding, None, None]:
    with RoleBinding(
        client=admin_client,
        namespace=model_registry_namespace,
        name=name,
        role_ref_name=mr_access_role.name,
        role_ref_kind=mr_access_role.kind,
        subjects_kind=subjects_kind,
        subjects_name=subjects_name,
    ) as mr_access_role_binding:
        yield mr_access_role_binding
