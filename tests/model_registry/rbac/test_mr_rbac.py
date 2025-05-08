import pytest
from typing import Self, Callable, ContextManager
import os
from simple_logger.logger import get_logger
from contextlib import nullcontext

from model_registry import ModelRegistry as ModelRegistryClient
from tests.model_registry.constants import MR_INSTANCE_NAME, MR_NAMESPACE
from tests.model_registry.rbac.utils import get_token, build_mr_client_args
from tests.model_registry.utils import get_endpoint_from_mr_service, get_mr_service_by_label
from kubernetes.dynamic import DynamicClient
from ocp_resources.model_registry import ModelRegistry
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from utilities.constants import DscComponents, Protocols
from mr_openapi.exceptions import ForbiddenException


LOGGER = get_logger(name=__name__)
NEW_GROUP_NAME = "test-model-registry-group"


def assert_positive_mr_registry(model_registry_instance, model_registry_namespace, admin_client, user_name, password):
    user_token = get_token(user_name=user_name, password=password, admin_client=admin_client)
    namespace_instance = admin_client.resources.get(api_version="v1", kind="Namespace").get(
        name=model_registry_namespace
    )
    svc = get_mr_service_by_label(client=admin_client, ns=namespace_instance, mr_instance=model_registry_instance)
    endpoint = get_endpoint_from_mr_service(service=svc, protocol=Protocols.REST)
    client_args = build_mr_client_args(rest_endpoint=endpoint, token=user_token, author="rbac-test-user-granted")
    mr_client = ModelRegistryClient(**client_args)
    assert mr_client is not None, "Client initialization failed after granting permissions"
    LOGGER.info("Client instantiated successfully after granting permissions.")


@pytest.mark.parametrize(
    "updated_dsc_component_state_scope_class",
    [
        pytest.param({
            "component_patch": {
                DscComponents.MODELREGISTRY: {
                    "managementState": DscComponents.ManagementState.MANAGED,
                    "registriesNamespace": MR_NAMESPACE,
                },
            }
        })
    ],
    indirect=True,
    scope="class",
)
@pytest.mark.usefixtures("updated_dsc_component_state_scope_class")
class TestUserPermission:
    """
    Test suite for verifying user and group permissions for the Model Registry.
    """

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "user_name, password, context_manager",
        [
            ("htpasswd-cluster-admin-user", os.environ.get("ADMIN_PASSWORD"), nullcontext()),
            ("ldap-user1", os.environ.get("NON_ADMIN_PASSWORD"), pytest.raises(ForbiddenException)),
        ],
    )
    def test_user_permission(
        self: Self,
        model_registry_instance: ModelRegistry,
        model_registry_namespace: str,
        admin_client: DynamicClient,
        user_name: str,
        password: str,
        context_manager: ContextManager,
    ):
        """
        Test that a user with permission can access the Model Registry,
        and a user without permission receives a ForbiddenException.
        """
        LOGGER.info("-----Test that a user with permission can access the Model Registry-----")
        assert model_registry_instance.name == MR_INSTANCE_NAME
        user_token = get_token(user_name=user_name, password=password, admin_client=admin_client)
        namespace_instance = admin_client.resources.get(api_version="v1", kind="Namespace").get(
            name=model_registry_namespace
        )
        svc = get_mr_service_by_label(client=admin_client, ns=namespace_instance, mr_instance=model_registry_instance)
        endpoint = get_endpoint_from_mr_service(service=svc, protocol=Protocols.REST)
        client_args = build_mr_client_args(rest_endpoint=endpoint, token=user_token, author="rbac-test")
        with context_manager as exc:
            _ = ModelRegistryClient(**client_args)

        if exc:
            http_error = exc.value
            assert http_error.body is not None, "HTTPError should have a response object"
            LOGGER.info(f"Received expected HTTP error: Status Code {http_error.status}")
            assert http_error.status == 403, f"Expected HTTP 403 Forbidden, but got {http_error.status}"
            LOGGER.info("Successfully received expected HTTP 403 status code.")
        else:  # If no exception was raised
            LOGGER.info("Successfully created Model Registry client")

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "user_name, password",
        [
            ("ldap-user1", os.environ.get("NON_ADMIN_PASSWORD")),
        ],
    )
    @pytest.mark.usefixtures("updated_dsc_component_state_scope_class", "user_in_group_context")
    def test_user_added_to_group(
        self: Self,
        model_registry_instance: ModelRegistry,
        model_registry_namespace: str,
        admin_client: DynamicClient,
        user_name: str,
        password: str,
        user_in_group_context: Callable[[str], ContextManager],
    ):
        """
        Test that a user cannot access the Model Registry before being added to a group,
        and can access it after being added to the group.
        """
        LOGGER.info(
            "-----Test that a user can access to the Model Registry once added to a "
            "group that has the permissions to access it-----"
        )
        assert model_registry_instance.name == MR_INSTANCE_NAME

        user_token = get_token(user_name=user_name, password=password, admin_client=admin_client)
        namespace_instance = admin_client.resources.get(api_version="v1", kind="Namespace").get(
            name=model_registry_namespace
        )
        svc = get_mr_service_by_label(client=admin_client, ns=namespace_instance, mr_instance=model_registry_instance)
        endpoint = get_endpoint_from_mr_service(service=svc, protocol=Protocols.REST)
        client_args = build_mr_client_args(rest_endpoint=endpoint, token=user_token, author="rbac-test-denied")

        LOGGER.info("User has no access to MR")
        with pytest.raises(ForbiddenException) as exc_info:
            _ = ModelRegistryClient(**client_args)

        http_error = exc_info.value
        assert http_error.body is not None, "HTTPError should have a response object"
        LOGGER.info(f"Received expected HTTP error: Status Code {http_error.status}")
        assert http_error.status == 403, f"Expected HTTP 403 Forbidden, but got {http_error.status}"
        LOGGER.info("Successfully received expected HTTP 403 status code.")

        LOGGER.info("Add user to the model registry users group")
        with user_in_group_context(user_name):
            assert_positive_mr_registry(
                model_registry_instance=model_registry_instance,
                model_registry_namespace=model_registry_namespace,
                admin_client=admin_client,
                user_name=user_name,
                password=password,
            )

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "user_name, password, new_group",
        [
            ("ldap-user1", os.environ.get("NON_ADMIN_PASSWORD"), (NEW_GROUP_NAME, "ldap-user1")),
        ],
        indirect=["new_group"],
    )
    @pytest.mark.usefixtures("updated_dsc_component_state_scope_class", "mr_access_role", "new_group")
    def test_create_group(
        self: Self,
        model_registry_instance: ModelRegistry,
        model_registry_namespace: str,
        admin_client: DynamicClient,
        user_name: str,
        password: str,
        mr_access_role: Role,
    ):
        """
        Test creating a group, granting it model registry access, and verifying user access.
        """
        LOGGER.info(
            "-----Test that a new group can be granted access to Model Registry and user added to it can access MR-----"
        )
        LOGGER.info("Group created and user added to it")

        with RoleBinding(
            client=admin_client,
            namespace=model_registry_namespace,
            name="test-model-registry-group-edit",
            role_ref_name=mr_access_role.name,
            role_ref_kind=mr_access_role.kind,
            subjects_kind="Group",
            subjects_name=NEW_GROUP_NAME,
        ):
            LOGGER.info("User should have access to MR after the group is granted edit access via a RoleBinding")
            assert_positive_mr_registry(
                model_registry_instance=model_registry_instance,
                model_registry_namespace=model_registry_namespace,
                admin_client=admin_client,
                user_name=user_name,
                password=password,
            )

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "user_name, password",
        [
            ("ldap-user1", os.environ.get("NON_ADMIN_PASSWORD")),
        ],
    )
    @pytest.mark.usefixtures("updated_dsc_component_state_scope_class", "mr_access_role")
    def test_add_single_user(
        self: Self,
        model_registry_instance: ModelRegistry,
        model_registry_namespace: str,
        admin_client: DynamicClient,
        user_name: str,
        password: str,
        mr_access_role: Role,
    ):
        """
        Test that adding a single user to the Model Registry's permitted list allows
        that user to access the Model Registry.
        """
        LOGGER.info(
            "-----Test that adding a single user to the Model Registry's permitted list allows "
            "that user to access the MR-----"
        )
        with RoleBinding(
            client=admin_client,
            namespace=model_registry_namespace,
            name="test-model-registry-access",
            role_ref_name=mr_access_role.name,
            role_ref_kind=mr_access_role.kind,
            subjects_kind="User",
            subjects_name=user_name,
        ):
            LOGGER.info("User should have access to MR after the RoleBinding for the user is created")
            assert_positive_mr_registry(
                model_registry_instance=model_registry_instance,
                model_registry_namespace=model_registry_namespace,
                admin_client=admin_client,
                user_name=user_name,
                password=password,
            )
