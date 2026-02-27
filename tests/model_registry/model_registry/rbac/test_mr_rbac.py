"""
Test suite for verifying user and group permissions for the Model Registry.

This suite tests various RBAC scenarios including:
- Basic user access permissions (admin vs normal user)
- Group-based access control
- User addition to groups and permission changes
- Role and RoleBinding management
"""

from collections.abc import Generator
from typing import Self

import pytest
from kubernetes.dynamic import DynamicClient
from model_registry import ModelRegistry as ModelRegistryClient
from mr_openapi.exceptions import ForbiddenException
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.deployment import Deployment
from ocp_resources.group import Group
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.role_binding import RoleBinding
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutSampler

from tests.model_registry.constants import NUM_MR_INSTANCES
from tests.model_registry.model_registry.rbac.multiple_instance_utils import MR_MULTIPROJECT_TEST_SCENARIO_PARAMS
from tests.model_registry.model_registry.rbac.utils import (
    assert_forbidden_access,
    assert_positive_mr_registry,
    build_mr_client_args,
    grant_mr_access,
    revoke_mr_access,
)
from tests.model_registry.utils import get_endpoint_from_mr_service, get_mr_service_by_label, get_mr_user_token
from utilities.constants import Protocols
from utilities.infra import get_openshift_token
from utilities.resources.model_registry_modelregistry_opendatahub_io import ModelRegistry
from utilities.user_utils import UserTestSession

LOGGER = get_logger(name=__name__)
pytestmark = [pytest.mark.usefixtures("original_user", "test_idp_user")]


@pytest.mark.usefixtures(
    "updated_dsc_component_state_scope_session",
    "model_registry_namespace",
    "model_registry_metadata_db_resources",
    "model_registry_instance",
)
@pytest.mark.custom_namespace
class TestUserPermission:
    @pytest.mark.tier1
    def test_user_permission_non_admin_user(
        self: Self,
        is_byoidc: bool,
        admin_client: DynamicClient,
        test_idp_user,
        model_registry_instance_rest_endpoint: list[str],
        user_credentials_rbac: dict[str, str],
        login_as_test_user: None,
    ):
        """
        This test verifies that non-admin users cannot access the Model Registry (403 Forbidden)
        """
        if is_byoidc:
            token = get_mr_user_token(admin_client=admin_client, user_credentials_rbac=user_credentials_rbac)
        else:
            token = get_openshift_token()

        client_args = build_mr_client_args(rest_endpoint=model_registry_instance_rest_endpoint[0], token=token)
        with pytest.raises(ForbiddenException) as exc_info:
            ModelRegistryClient(**client_args)
        assert exc_info.value.status == 403, f"Expected HTTP 403 ForbiddenException, but got {exc_info.value.status}"
        LOGGER.info("Successfully received expected HTTP 403 status code")

    @pytest.mark.tier1
    def test_user_added_to_group(
        self: Self,
        is_byoidc: bool,
        admin_client: DynamicClient,
        model_registry_instance_rest_endpoint: list[str],
        test_idp_user: UserTestSession,
        user_credentials_rbac: dict[str, str],
        model_registry_group_with_user: Group,
        login_as_test_user: Generator[UserTestSession],
    ):
        """
        This test verifies that:
        1. After adding the user to the appropriate group, they gain access
        """
        # Wait for access to be granted
        # Create a copy to avoid mutating the shared fixture
        creds_copy = user_credentials_rbac.copy()
        creds_copy["username"] = "mr-user1"
        sampler = TimeoutSampler(
            wait_timeout=240,
            sleep=5,
            func=assert_positive_mr_registry,
            model_registry_instance_rest_endpoint=model_registry_instance_rest_endpoint[0],
            token=get_openshift_token()
            if not is_byoidc
            else get_mr_user_token(admin_client=admin_client, user_credentials_rbac=creds_copy),
        )
        for _ in sampler:
            break  # Break after first successful iteration
        LOGGER.info("Successfully accessed Model Registry")

    @pytest.mark.tier1
    def test_create_group(
        self: Self,
        skip_test_on_byoidc: None,
        test_idp_user: UserTestSession,
        model_registry_instance_rest_endpoint: list[str],
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
            model_registry_instance_rest_endpoint=model_registry_instance_rest_endpoint[0],
        )

    @pytest.mark.tier1
    def test_add_single_user_role_binding(
        self: Self,
        is_byoidc: bool,
        admin_client: DynamicClient,
        test_idp_user: UserTestSession,
        model_registry_instance_rest_endpoint: list[str],
        user_credentials_rbac: dict[str, str],
        created_role_binding_user: RoleBinding,
        login_as_test_user: None,
    ):
        """
        Test granting Model Registry access to a single user.

        This test verifies that:
        1. A single user can be granted Model Registry access via RoleBinding
        2. The user can access the Model Registry after being granted access
        """
        if is_byoidc:
            # Create a copy to avoid mutating the shared fixture
            creds_copy = user_credentials_rbac.copy()
            creds_copy["username"] = "mr-non-admin"
            sampler = TimeoutSampler(
                wait_timeout=120,
                sleep=5,
                func=assert_positive_mr_registry,
                model_registry_instance_rest_endpoint=model_registry_instance_rest_endpoint[0],
                token=get_mr_user_token(admin_client=admin_client, user_credentials_rbac=creds_copy),
            )
            for _ in sampler:
                break  # Break after first successful iteration
            LOGGER.info("Successfully accessed Model Registry")
        else:
            assert_positive_mr_registry(model_registry_instance_rest_endpoint=model_registry_instance_rest_endpoint[0])


class TestUserMultiProjectPermission:
    """
    Test suite for verifying user permissions in a multi-project setup for the Model Registry.
    """

    @pytest.mark.parametrize(
        (
            "db_secret_parametrized, "
            "db_pvc_parametrized, "
            "db_service_parametrized, "
            "db_deployment_parametrized, "
            "model_registry_instance_parametrized"
        ),
        MR_MULTIPROJECT_TEST_SCENARIO_PARAMS,
        indirect=True,
    )
    @pytest.mark.tier1
    def test_user_permission_multi_project_parametrized(
        self: Self,
        is_byoidc: bool,
        test_idp_user: UserTestSession,
        admin_client: DynamicClient,
        updated_dsc_component_state_scope_session: DataScienceCluster,
        model_registry_namespace: str,
        db_secret_parametrized: list[Secret],
        db_pvc_parametrized: list[PersistentVolumeClaim],
        db_service_parametrized: list[Service],
        db_deployment_parametrized: list[Deployment],
        user_credentials_rbac: dict[str, str],
        model_registry_instance_parametrized: list[ModelRegistry],
        login_as_test_user: None,
    ):
        """
        Verify that a user can be granted access to one MR instance at a time.
        All resources (MR instances and databases) are created in the same dynamically generated namespace.
        """
        if len(model_registry_instance_parametrized) != NUM_MR_INSTANCES:
            raise ValueError(
                f"Expected {NUM_MR_INSTANCES} MR instances, but got {len(model_registry_instance_parametrized)}"
            )

        LOGGER.info(f"Model Registry namespace: {model_registry_namespace}")

        # Prepare MR instances and endpoints
        mr_data = []
        for mr_instance in model_registry_instance_parametrized:
            service = get_mr_service_by_label(
                client=admin_client,
                namespace_name=model_registry_namespace,
                mr_instance=mr_instance,
            )
            endpoint = get_endpoint_from_mr_service(svc=service, protocol=Protocols.REST)
            mr_data.append({"instance": mr_instance, "endpoint": endpoint, "name": mr_instance.name})

        token = (
            get_openshift_token()
            if not is_byoidc
            else (get_mr_user_token(admin_client=admin_client, user_credentials_rbac=user_credentials_rbac))
        )

        # Test each MR instance sequentially
        for i, current_mr_data in enumerate(mr_data):
            current_mr = current_mr_data["instance"]
            current_endpoint = current_mr_data["endpoint"]

            LOGGER.info(f"Testing access to MR instance {i + 1}/{len(mr_data)}: {current_mr.name}")

            # Grant access to current instance
            grant_mr_access(
                admin_client=admin_client,
                user=user_credentials_rbac["username"],
                mr_instance_name=current_mr.name,
                model_registry_namespace=model_registry_namespace,
            )

            # Verify access to current instance
            sampler = TimeoutSampler(
                wait_timeout=240,
                sleep=5,
                func=assert_positive_mr_registry,
                model_registry_instance_rest_endpoint=current_endpoint,
                token=token,
            )
            for _ in sampler:
                break

            # Verify NO access to other instances
            other_mr_names = [mr["name"] for j, mr in enumerate(mr_data) if j != i]
            for j, other_mr_data in enumerate(mr_data):
                if i != j:
                    # Wait for role reconciliation - retry until ForbiddenException is raised
                    sampler = TimeoutSampler(
                        wait_timeout=360,
                        sleep=10,
                        func=assert_forbidden_access,
                        endpoint=other_mr_data["endpoint"],
                        token=token,
                    )
                    for _ in sampler:
                        break

            LOGGER.info(f"User has access to {current_mr.name}, but not to: {', '.join(other_mr_names)}")

            # Revoke access (except for the last instance)
            if i < len(mr_data) - 1:
                revoke_mr_access(
                    admin_client=admin_client,
                    user=user_credentials_rbac["username"],
                    mr_instance_name=current_mr.name,
                    model_registry_namespace=model_registry_namespace,
                )

        # Clean up - revoke access from the last instance
        revoke_mr_access(
            admin_client=admin_client,
            user=user_credentials_rbac["username"],
            mr_instance_name=mr_data[-1]["instance"].name,
            model_registry_namespace=model_registry_namespace,
        )
