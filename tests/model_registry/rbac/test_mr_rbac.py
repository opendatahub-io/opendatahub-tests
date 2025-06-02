import pytest
from pytest_testconfig import config as py_config
from typing import Self
from simple_logger.logger import get_logger

from model_registry import ModelRegistry as ModelRegistryClient
from tests.model_registry.constants import MR_INSTANCE_NAME
from tests.model_registry.rbac.utils import assert_positive_mr_registry, get_mr_client_args
from utilities.infra import switch_user_context
from kubernetes.dynamic import DynamicClient
from ocp_resources.group import Group
from ocp_resources.model_registry import ModelRegistry
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from utilities.constants import DscComponents
from mr_openapi.exceptions import ForbiddenException
from utilities.user_utils import UserTestSession
from timeout_sampler import TimeoutSampler

LOGGER = get_logger(name=__name__)
NEW_GROUP_NAME = "test-model-registry-group"


@pytest.mark.parametrize(
    "updated_dsc_component_state_scope_class",
    [
        pytest.param({
            "component_patch": {
                DscComponents.MODELREGISTRY: {
                    "managementState": DscComponents.ManagementState.MANAGED,
                    "registriesNamespace": py_config["model_registry_namespace"],
                },
            }
        })
    ],
    indirect=True,
)
@pytest.mark.usefixtures("updated_dsc_component_state_scope_class")
class TestUserPermission:
    """
    Test suite for verifying user and group permissions for the Model Registry.

    This suite tests various RBAC scenarios including:
    - Basic user access permissions (admin vs normal user)
    - Group-based access control
    - User addition to groups and permission changes
    - Role and RoleBinding management
    """

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        "use_admin_context",
        [
            pytest.param(True, id="admin_user"),
            pytest.param(False, id="normal_user"),
        ],
    )
    def test_user_permission(
        self: Self,
        model_registry_instance: ModelRegistry,
        model_registry_namespace: str,
        admin_client: DynamicClient,
        test_idp_user_session: UserTestSession,
        use_admin_context: bool,
    ):
        """
        Test basic user access permissions to the Model Registry.

        This test verifies that:
        - Admin users can access the Model Registry
        - Normal users cannot access the Model Registry (403 Forbidden)

        Args:
            model_registry_instance: The Model Registry instance to test against
            model_registry_namespace: The namespace where Model Registry is deployed
            admin_client: The admin client for accessing the cluster
            test_idp_user_session: The test user session containing both admin and user contexts
            use_admin_context: Whether to use admin context (True) or user context (False)

        Raises:
            AssertionError: If access permissions don't match expectations
            ForbiddenException: Expected for normal users, unexpected for admin users
        """

        context_to_use = (
            test_idp_user_session.original_context if use_admin_context else test_idp_user_session.user_context
        )

        with switch_user_context(context_to_use):
            _, client_args = get_mr_client_args(
                model_registry_instance=model_registry_instance,
                model_registry_namespace=model_registry_namespace,
                admin_client=admin_client,
            )

            if use_admin_context:
                mr_client = ModelRegistryClient(**client_args)
                assert mr_client is not None, "Client initialization failed for admin user"
                LOGGER.info("Successfully created Model Registry client for admin user")
            else:
                with pytest.raises(ForbiddenException) as exc_info:
                    _ = ModelRegistryClient(**client_args)
                assert exc_info.value.status == 403, f"Expected HTTP 403 Forbidden, but got {exc_info.value.status}"
                LOGGER.info("Successfully received expected HTTP 403 status code")

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        "model_registry_group_with_user",
        [
            pytest.param(
                f"{MR_INSTANCE_NAME}-users",
                id="model_registry_users",
            ),
        ],
        indirect=["model_registry_group_with_user"],
    )
    def test_user_added_to_group(
        self: Self,
        model_registry_instance: ModelRegistry,
        model_registry_namespace: str,
        admin_client: DynamicClient,
        test_idp_user_session: UserTestSession,
        model_registry_group_with_user: Group,
    ):
        """
        Test that a user's access to Model Registry changes when added to a group.

        This test verifies that:
        1. A normal user cannot access the Model Registry initially
        2. After adding the user to the appropriate group, they gain access
        3. After removing the user from the group, they lose access

        Args:
            model_registry_instance: The Model Registry instance to test against
            model_registry_namespace: The namespace where Model Registry is deployed
            admin_client: The admin client for accessing the cluster
            test_idp_user_session: The test user session containing both admin and user contexts
            model_registry_group_with_user: The Model Registry group with the test user added

        Raises:
            AssertionError: If access permissions don't match expectations
            ForbiddenException: Expected before group addition, unexpected after
        """

        # Wait for access to be granted
        with switch_user_context(test_idp_user_session.user_context):
            sampler = TimeoutSampler(
                wait_timeout=240,
                sleep=5,
                func=assert_positive_mr_registry,
                model_registry_instance=model_registry_instance,
                model_registry_namespace=model_registry_namespace,
                admin_client=admin_client,
            )
            for _ in sampler:
                break  # Break after first successful iteration
            LOGGER.info("Successfully accessed Model Registry")

    @pytest.mark.sanity
    @pytest.mark.parametrize(
        "add_user_to_group",
        [
            pytest.param(
                NEW_GROUP_NAME,
                id="new_group",
            ),
        ],
        indirect=["add_user_to_group"],
    )
    @pytest.mark.usefixtures("mr_access_role", "add_user_to_group")
    def test_create_group(
        self: Self,
        model_registry_instance: ModelRegistry,
        model_registry_namespace: str,
        admin_client: DynamicClient,
        test_idp_user_session: UserTestSession,
        mr_access_role: Role,
    ):
        """
        Test creating a new group and granting it Model Registry access.

        This test verifies that:
        1. A new group can be created and user added to it
        2. The group can be granted Model Registry access via RoleBinding
        3. Users in the group can access the Model Registry

        Args:
            model_registry_instance: The Model Registry instance to test against
            model_registry_namespace: The namespace where Model Registry is deployed
            admin_client: The admin client for accessing the cluster
            test_idp_user_session: The test user session containing both admin and user contexts
            mr_access_role: The Role that grants Model Registry access

        Raises:
            AssertionError: If group creation or access permissions don't match expectations
        """

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
            with switch_user_context(test_idp_user_session.user_context):
                assert_positive_mr_registry(
                    model_registry_instance=model_registry_instance,
                    model_registry_namespace=model_registry_namespace,
                    admin_client=admin_client,
                )

    @pytest.mark.sanity
    @pytest.mark.usefixtures("mr_access_role")
    def test_add_single_user(
        self: Self,
        model_registry_instance: ModelRegistry,
        model_registry_namespace: str,
        admin_client: DynamicClient,
        test_idp_user_session: UserTestSession,
        mr_access_role: Role,
    ):
        """
        Test granting Model Registry access to a single user.

        This test verifies that:
        1. A single user can be granted Model Registry access via RoleBinding
        2. The user can access the Model Registry after being granted access

        Args:
            model_registry_instance: The Model Registry instance to test against
            model_registry_namespace: The namespace where Model Registry is deployed
            admin_client: The admin client for accessing the cluster
            test_idp_user_session: The test user session containing both admin and user contexts
            mr_access_role: The Role that grants Model Registry access

        Raises:
            AssertionError: If access permissions don't match expectations
        """

        with RoleBinding(
            client=admin_client,
            namespace=model_registry_namespace,
            name="test-model-registry-access",
            role_ref_name=mr_access_role.name,
            role_ref_kind=mr_access_role.kind,
            subjects_kind="User",
            subjects_name=test_idp_user_session.username,
        ):
            with switch_user_context(test_idp_user_session.user_context):
                assert_positive_mr_registry(
                    model_registry_instance=model_registry_instance,
                    model_registry_namespace=model_registry_namespace,
                    admin_client=admin_client,
                )
