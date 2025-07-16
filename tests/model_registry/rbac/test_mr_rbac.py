"""
Test suite for verifying user and group permissions for the Model Registry.

This suite tests various RBAC scenarios including:
- Basic user access permissions (admin vs normal user)
- Group-based access control
- User addition to groups and permission changes
- Role and RoleBinding management
"""

import pytest
from pytest_testconfig import config as py_config
from typing import Self, Generator
from simple_logger.logger import get_logger

from model_registry import ModelRegistry as ModelRegistryClient
from timeout_sampler import TimeoutSampler

from ocp_resources.group import Group
from ocp_resources.role_binding import RoleBinding
from tests.model_registry.rbac.utils import build_mr_client_args, assert_positive_mr_registry
from utilities.infra import get_openshift_token
from utilities.constants import DscComponents
from mr_openapi.exceptions import ForbiddenException
from utilities.user_utils import UserTestSession

LOGGER = get_logger(name=__name__)
pytestmark = [pytest.mark.usefixtures("original_user", "test_idp_user")]


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
        }),
    ],
    indirect=True,
)
@pytest.mark.usefixtures(
    "updated_dsc_component_state_scope_class",
    "is_model_registry_oauth",
    "model_registry_mysql_metadata_db",
    "model_registry_instance_mysql",
)
class TestUserPermission:
    @pytest.mark.sanity
    def test_user_permission_non_admin_user(
        self: Self,
        test_idp_user,
        model_registry_instance_rest_endpoint: str,
        login_as_test_user: None,
    ):
        """
        This test verifies that non-admin users cannot access the Model Registry (403 Forbidden)
        """
        client_args = build_mr_client_args(
            rest_endpoint=model_registry_instance_rest_endpoint, token=get_openshift_token()
        )
        with pytest.raises(ForbiddenException) as exc_info:
            ModelRegistryClient(**client_args)
        assert exc_info.value.status == 403, f"Expected HTTP 403 Forbidden, but got {exc_info.value.status}"
        LOGGER.info("Successfully received expected HTTP 403 status code")

    @pytest.mark.sanity
    def test_user_added_to_group(
        self: Self,
        model_registry_instance_rest_endpoint: str,
        test_idp_user: UserTestSession,
        model_registry_group_with_user: Group,
        login_as_test_user: Generator[UserTestSession, None, None],
    ):
        """
        This test verifies that:
        1. After adding the user to the appropriate group, they gain access
        """
        # Wait for access to be granted
        sampler = TimeoutSampler(
            wait_timeout=240,
            sleep=5,
            func=assert_positive_mr_registry,
            model_registry_instance_rest_endpoint=model_registry_instance_rest_endpoint,
            token=get_openshift_token(),
        )
        for _ in sampler:
            break  # Break after first successful iteration
        LOGGER.info("Successfully accessed Model Registry")

    @pytest.mark.sanity
    def test_create_group(
        self: Self,
        test_idp_user: UserTestSession,
        model_registry_instance_rest_endpoint: str,
        created_role_binding_group: RoleBinding,
        login_as_test_user: None,
    ):
        """
        Test creating a new group and granting it Model Registry access.

        This test verifies that:
        1. A new group can be created and user added to it
        2. The group can be granted Model Registry access via RoleBinding
        3. Users in the group can access the Model Registry
        """
        assert_positive_mr_registry(
            model_registry_instance_rest_endpoint=model_registry_instance_rest_endpoint,
        )

    @pytest.mark.sanity
    def test_add_single_user_role_binding(
        self: Self,
        test_idp_user: UserTestSession,
        model_registry_instance_rest_endpoint: str,
        created_role_binding_user: RoleBinding,
        login_as_test_user: None,
    ):
        """
        Test granting Model Registry access to a single user.

        This test verifies that:
        1. A single user can be granted Model Registry access via RoleBinding
        2. The user can access the Model Registry after being granted access
        """
        assert_positive_mr_registry(
            model_registry_instance_rest_endpoint=model_registry_instance_rest_endpoint,
        )
