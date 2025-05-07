import pytest
from typing import Self, Callable, ContextManager
import os
from simple_logger.logger import get_logger
from contextlib import nullcontext


from tests.model_registry.constants import MR_INSTANCE_NAME, MR_NAMESPACE
from tests.model_registry.rbac.utils import get_token, assert_mr_client
from kubernetes.dynamic import DynamicClient
from ocp_resources.model_registry import ModelRegistry
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from utilities.constants import DscComponents
from mr_openapi.exceptions import ForbiddenException


LOGGER = get_logger(name=__name__)
NEW_GROUP_NAME = "test-model-registry-group"


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
)
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
    @pytest.mark.usefixtures("updated_dsc_component_state_scope_class")
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

        assert_mr_client(
            user_token=user_token,
            admin_client=admin_client,
            context=context_manager,
            mr_instance=model_registry_instance,
            mr_namespace_name=model_registry_namespace,
        )

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

        LOGGER.info("User has no access to MR")
        assert_mr_client(
            user_token=user_token,
            admin_client=admin_client,
            context=pytest.raises(ForbiddenException),
            mr_instance=model_registry_instance,
            mr_namespace_name=model_registry_namespace,
        )

        LOGGER.info("Add user to the model registry users group")
        with user_in_group_context(user_name):
            user_token = get_token(user_name=user_name, password=password, admin_client=admin_client)

            LOGGER.info("User has access to MR")
            assert_mr_client(
                user_token=user_token,
                admin_client=admin_client,
                context=nullcontext(),
                mr_instance=model_registry_instance,
                mr_namespace_name=model_registry_namespace,
            )

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "user_name, password, new_group",
        [
            ("ldap-user1", os.environ.get("NON_ADMIN_PASSWORD"), (NEW_GROUP_NAME, "ldap-user1")),
        ],
        indirect=["new_group"],
    )
    @pytest.mark.usefixtures("updated_dsc_component_state_scope_class", "model_registry_role", "new_group")
    def test_create_group(
        self: Self,
        model_registry_instance: ModelRegistry,
        model_registry_namespace: str,
        admin_client: DynamicClient,
        user_name: str,
        password: str,
        model_registry_role: Role,
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
            role_ref_name=model_registry_role.name,
            role_ref_kind=model_registry_role.kind,
            subjects_kind="Group",
            subjects_name=NEW_GROUP_NAME,
        ):
            LOGGER.info("User should have access to MR after the group is granted edit access via a RoleBinding")
            user_token = get_token(user_name=user_name, password=password, admin_client=admin_client)
            assert_mr_client(
                user_token=user_token,
                admin_client=admin_client,
                context=nullcontext(),
                mr_instance=model_registry_instance,
                mr_namespace_name=model_registry_namespace,
            )

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "user_name, password",
        [
            ("ldap-user1", os.environ.get("NON_ADMIN_PASSWORD")),
        ],
    )
    @pytest.mark.usefixtures("updated_dsc_component_state_scope_class", "model_registry_role")
    def test_add_single_user(
        self: Self,
        model_registry_instance: ModelRegistry,
        model_registry_namespace: str,
        admin_client: DynamicClient,
        user_name: str,
        password: str,
        model_registry_role: Role,
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
            role_ref_name=model_registry_role.name,
            role_ref_kind=model_registry_role.kind,
            subjects_kind="User",
            subjects_name=user_name,
        ):
            LOGGER.info("User should have access to MR after the RoleBinding for the user is created")
            user_token = get_token(user_name=user_name, password=password, admin_client=admin_client)
            assert_mr_client(
                user_token=user_token,
                admin_client=admin_client,
                context=nullcontext(),
                mr_instance=model_registry_instance,
                mr_namespace_name=model_registry_namespace,
            )
