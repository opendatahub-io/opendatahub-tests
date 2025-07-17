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
from ocp_resources.secret import Secret
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.service import Service
from ocp_resources.deployment import Deployment
from tests.model_registry.rbac.utils import build_mr_client_args, assert_positive_mr_registry, assert_forbidden_access
from utilities.infra import get_openshift_token
from utilities.constants import DscComponents
from mr_openapi.exceptions import ForbiddenException
from utilities.user_utils import UserTestSession
from kubernetes.dynamic import DynamicClient
from ocp_resources.model_registry_modelregistry_opendatahub_io import ModelRegistry

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
@pytest.mark.usefixtures("updated_dsc_component_state_scope_class")
class TestUserMultiProjectPermission:
    """
    Test suite for verifying user permissions in a multi-project setup for the Model Registry.
    """

    def test_user_permission_multi_project(
        self: Self,
        test_idp_user: UserTestSession,
        admin_client: DynamicClient,
        model_registry_namespace: str,
        db_secret_1: Secret,
        db_pvc_1: PersistentVolumeClaim,
        db_service_1: Service,
        db_deployment_1: Deployment,
        model_registry_instance_1: ModelRegistry,
        db_secret_2: Secret,
        db_pvc_2: PersistentVolumeClaim,
        db_service_2: Service,
        db_deployment_2: Deployment,
        model_registry_instance_2: ModelRegistry,
        login_as_test_user: None,
    ):
        """
        Verify that a user can be granted access to one MR instance at a time.
        """
        from tests.model_registry.utils import get_mr_service_by_label, get_endpoint_from_mr_service
        from tests.model_registry.rbac.utils import grant_mr_access, revoke_mr_access
        from utilities.constants import Protocols

        # Get endpoints for both MR instances
        service1 = get_mr_service_by_label(
            client=admin_client,
            namespace_name=model_registry_namespace,
            mr_instance=model_registry_instance_1,
        )
        endpoint1 = get_endpoint_from_mr_service(svc=service1, protocol=Protocols.REST)

        service2 = get_mr_service_by_label(
            client=admin_client,
            namespace_name=model_registry_namespace,
            mr_instance=model_registry_instance_2,
        )
        endpoint2 = get_endpoint_from_mr_service(svc=service2, protocol=Protocols.REST)

        # 1. Grant access to the first instance and verify
        grant_mr_access(
            admin_client=admin_client,
            user=test_idp_user.username,
            mr_instance_name=model_registry_instance_1.name,
            model_registry_namespace=model_registry_namespace,
        )
        sampler = TimeoutSampler(
            wait_timeout=240,
            sleep=5,
            func=assert_positive_mr_registry,
            model_registry_instance_rest_endpoint=endpoint1,
            token=get_openshift_token(),
        )
        for _ in sampler:
            break
        with pytest.raises(ForbiddenException):
            ModelRegistryClient(**build_mr_client_args(rest_endpoint=endpoint2, token=get_openshift_token()))

        LOGGER.info(f"User has access to {model_registry_instance_1.name}, but not {model_registry_instance_2.name}")

        # 2. Revoke access from the first, grant to the second, and verify
        revoke_mr_access(
            admin_client=admin_client,
            user=test_idp_user.username,
            mr_instance_name=model_registry_instance_1.name,
            model_registry_namespace=model_registry_namespace,
        )
        grant_mr_access(
            admin_client=admin_client,
            user=test_idp_user.username,
            mr_instance_name=model_registry_instance_2.name,
            model_registry_namespace=model_registry_namespace,
        )

        sampler = TimeoutSampler(
            wait_timeout=240,
            sleep=5,
            func=assert_positive_mr_registry,
            model_registry_instance_rest_endpoint=endpoint2,
            token=get_openshift_token(),
        )
        for _ in sampler:
            break
        # Wait for role reconciliation - retry until ForbiddenException is raised
        sampler = TimeoutSampler(
            wait_timeout=240,
            sleep=5,
            func=assert_forbidden_access,
            endpoint=endpoint1,
            token=get_openshift_token(),
        )
        for _ in sampler:
            break

        LOGGER.info(
            f"User now has access to {model_registry_instance_2.name}, but not {model_registry_instance_1.name}"
        )

        revoke_mr_access(
            admin_client=admin_client,
            user=test_idp_user.username,
            mr_instance_name=model_registry_instance_2.name,
            model_registry_namespace=model_registry_namespace,
        )
