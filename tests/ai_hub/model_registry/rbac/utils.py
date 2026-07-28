import logging
from collections.abc import Generator
from typing import Any

from kubernetes.dynamic import DynamicClient
from model_registry import ModelRegistry as ModelRegistryClient
from mr_openapi.exceptions import ForbiddenException

from utilities.constants import Protocols
from utilities.infra import get_openshift_token
from utilities.openshift_resources.role import Role
from utilities.openshift_resources.role_binding import RoleBinding

LOGGER = logging.getLogger(__name__)


def build_mr_client_args(rest_endpoint: tuple[str, int], token: str, author: str = "rbac-test") -> dict[str, Any]:
    """Builds arguments for ModelRegistryClient.

    Args:
        rest_endpoint: Tuple of (address, port) from get_endpoint_from_mr_service.
        token: The token for the user.
        author: The author of the request.
    """
    address, port = rest_endpoint
    host, _, path = address.partition("/")
    server_url = f"{Protocols.HTTPS}://{host}:{port}/{path}" if path else f"{Protocols.HTTPS}://{host}:{port}"
    return {
        "server_address": server_url,
        "port": port,
        "user_token": token,
        "is_secure": False,
        "author": author,
    }


def assert_positive_mr_registry(
    model_registry_instance_rest_endpoint: tuple[str, int],
    token: str = "",
) -> None:
    """Assert that a user has access to the Model Registry.

    Args:
        model_registry_instance_rest_endpoint: Tuple of (address, port).
        token: user token.
    """
    client_args = build_mr_client_args(
        rest_endpoint=model_registry_instance_rest_endpoint,
        token=token or get_openshift_token(),
        author="rbac-test-user-granted",
    )
    mr_client = ModelRegistryClient(**client_args)
    assert mr_client is not None, "Client initialization failed after granting permissions"
    LOGGER.info("Client instantiated successfully after granting permissions.")


def create_role_binding(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    mr_access_role: Role,
    name: str,
    subjects_kind: str,
    subjects_name: str,
) -> Generator[RoleBinding]:
    with RoleBinding(
        namespace=model_registry_namespace,
        name=name,
        role_ref={
            "kind": mr_access_role.kind,
            "name": mr_access_role.name,
            "apiGroup": "rbac.authorization.k8s.io",
        },
        subjects=[
            {
                "kind": subjects_kind,
                "name": subjects_name,
                "apiGroup": "rbac.authorization.k8s.io",
            }
        ],
    ) as mr_access_role_binding:
        yield mr_access_role_binding


def grant_mr_access(
    admin_client: DynamicClient, user: str, mr_instance_name: str, model_registry_namespace: str
) -> tuple[Role, RoleBinding]:
    """Grant a user access to a Model Registry instance."""
    role_rules: list[dict[str, Any]] = [
        {
            "apiGroups": [""],
            "resources": ["services"],
            "resourceNames": [mr_instance_name],  # Grant access only to the specific MR service object
            "verbs": ["get"],
        }
    ]
    role_labels = {
        "app.kubernetes.io/component": "model-registry-test-rbac-multitenancy",
    }
    role = Role(
        name=f"{user}-{mr_instance_name}-role",
        namespace=model_registry_namespace,
        rules=role_rules,
        label=role_labels,
    )
    role.create()
    rb = RoleBinding(
        namespace=model_registry_namespace,
        name=f"{user}-{mr_instance_name}-access",
        role_ref={
            "kind": "Role",
            "name": f"{user}-{mr_instance_name}-role",
            "apiGroup": "rbac.authorization.k8s.io",
        },
        subjects=[
            {
                "kind": "User",
                "name": user,
                "apiGroup": "rbac.authorization.k8s.io",
            }
        ],
    )
    rb.create()
    LOGGER.info(f"Role {role.name} created successfully.")
    LOGGER.info(f"RoleBinding {rb.name} created successfully.")
    return role, rb


def revoke_mr_access(
    admin_client: DynamicClient, user: str, mr_instance_name: str, model_registry_namespace: str
) -> None:
    """Revoke a user's access to a Model Registry instance."""
    rb = RoleBinding(
        client=admin_client,
        namespace=model_registry_namespace,
        name=f"{user}-{mr_instance_name}-access",
    )
    rb.delete(wait=True)
    role = Role(
        client=admin_client,
        namespace=model_registry_namespace,
        name=f"{user}-{mr_instance_name}-role",
    )
    role.delete(wait=True)
    LOGGER.info(f"Role {role.name} deleted successfully.")
    LOGGER.info(f"RoleBinding {rb.name} deleted successfully.")


def assert_forbidden_access(endpoint: tuple[str, int], token: str) -> None:
    """Assert that access is properly forbidden."""
    try:
        ModelRegistryClient(**build_mr_client_args(rest_endpoint=endpoint, token=token))
        # If no exception is raised, the access is still granted - raise an error to continue retrying
        raise AssertionError("Access should be forbidden but client creation succeeded")
    except ForbiddenException:
        # This is what we want - access is properly forbidden
        pass
