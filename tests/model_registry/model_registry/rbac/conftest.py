from contextlib import ExitStack

import pytest
from typing import Generator, List, Any

from _pytest.fixtures import FixtureRequest
from simple_logger.logger import get_logger

from ocp_resources.deployment import Deployment
from utilities.resources.model_registry_modelregistry_opendatahub_io import ModelRegistry
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.role_binding import RoleBinding
from ocp_resources.role import Role
from ocp_resources.group import Group

from ocp_resources.resource import ResourceEditor
from kubernetes.dynamic import DynamicClient

from tests.model_registry.model_registry.rbac.utils import create_role_binding
from utilities.user_utils import UserTestSession
from tests.model_registry.model_registry.rbac.group_utils import create_group
from tests.model_registry.constants import (
    MR_INSTANCE_NAME,
    KUBERBACPROXY_STR,
)

LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="function")
def add_user_to_group(
    admin_client: DynamicClient,
    test_idp_user: UserTestSession,
) -> Generator[str, None, None]:
    """
    Fixture to create a group and add a test user to it.
    Uses create_group context manager to ensure proper cleanup.

    Args:
        admin_client: The admin client for accessing the cluster
        test_idp_user_session: The test user session containing user information

    Yields:
        str: The name of the created group
    """
    group_name = "test-model-registry-group"
    with create_group(
        admin_client=admin_client,
        group_name=group_name,
        users=[test_idp_user.username],
    ) as group_name:
        yield group_name


@pytest.fixture(scope="function")
def model_registry_group_with_user(
    is_byoidc: bool,
    admin_client: DynamicClient,
    test_idp_user: UserTestSession,
) -> Generator[Group, None, None]:
    """
    Fixture to manage a test user in a specified group.
    Adds the user to the group before the test, then removes them after.

    Args:
        admin_client: The admin client for accessing the cluster
        test_idp_user_session: The test user session containing user information

    Yields:
        Group: The group with the test user added
    """
    if is_byoidc:
        # this is no op. byoidc already has a group with user model-registry-user
        yield
    else:
        group_name = f"{MR_INSTANCE_NAME}-users"
        group = Group(
            client=admin_client,
            name=group_name,
            wait_for_resource=True,
        )

        # Add user to group
        with ResourceEditor(
            patches={
                group: {
                    "metadata": {"name": group_name},
                    "users": [test_idp_user.username],
                }
            }
        ) as _:
            LOGGER.info(f"Added user {test_idp_user.username} to {group_name} group")
            yield group


@pytest.fixture(scope="function")
def created_role_binding_group(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    mr_access_role: Role,
    test_idp_user: UserTestSession,
    add_user_to_group: str,
) -> Generator[RoleBinding, None, None]:
    yield from create_role_binding(
        admin_client=admin_client,
        model_registry_namespace=model_registry_namespace,
        name="test-model-registry-group-edit",
        mr_access_role=mr_access_role,
        subjects_kind="Group",
        subjects_name=add_user_to_group,
    )


@pytest.fixture(scope="function")
def created_role_binding_user(
    is_byoidc: bool,
    admin_client: DynamicClient,
    model_registry_namespace: str,
    mr_access_role: Role,
    user_credentials_rbac: dict[str, str],
) -> Generator[RoleBinding, None, None]:
    # Determine the username to use without mutating the shared fixture
    username = "mr-non-admin" if is_byoidc else user_credentials_rbac["username"]
    LOGGER.info(f"Using user {username}")
    yield from create_role_binding(
        admin_client=admin_client,
        model_registry_namespace=model_registry_namespace,
        name="test-model-registry-access",
        mr_access_role=mr_access_role,
        subjects_kind="User",
        subjects_name=username,
    )


# =============================================================================
# RESOURCE FIXTURES PARMETRIZED
# =============================================================================
@pytest.fixture(scope="class")
def db_secret_parametrized(
    request: FixtureRequest, admin_client: DynamicClient, teardown_resources: bool
) -> Generator[List[Secret], Any, Any]:
    """Create DB Secret parametrized"""
    with ExitStack() as stack:
        secrets = [
            stack.enter_context(
                Secret(
                    **param,
                    client=admin_client,
                    teardown=teardown_resources,
                )
            )
            for param in request.param
        ]
        yield secrets


@pytest.fixture(scope="class")
def db_pvc_parametrized(
    request: FixtureRequest, admin_client: DynamicClient, teardown_resources: bool
) -> Generator[List[PersistentVolumeClaim], Any, Any]:
    """Create DB PVC parametrized"""
    with ExitStack() as stack:
        pvc = [
            stack.enter_context(
                PersistentVolumeClaim(
                    **param,
                    client=admin_client,
                    teardown=teardown_resources,
                )
            )
            for param in request.param
        ]
        yield pvc


@pytest.fixture(scope="class")
def db_service_parametrized(
    request: FixtureRequest, admin_client: DynamicClient, teardown_resources: bool
) -> Generator[List[Service], Any, Any]:
    """Create DB Service parametrized"""
    with ExitStack() as stack:
        services = [
            stack.enter_context(
                Service(
                    **param,
                    client=admin_client,
                    teardown=teardown_resources,
                )
            )
            for param in request.param
        ]
        yield services


@pytest.fixture(scope="class")
def db_deployment_parametrized(
    request: FixtureRequest, admin_client: DynamicClient, teardown_resources: bool
) -> Generator[List[Deployment], Any, Any]:
    """Create DB Deployment parametrized"""
    with ExitStack() as stack:
        deployments = [
            stack.enter_context(
                Deployment(
                    **param,
                    client=admin_client,
                    teardown=teardown_resources,
                )
            )
            for param in request.param
        ]

        for deployment in deployments:
            deployment.wait_for_replicas(deployed=True)

        yield deployments


@pytest.fixture(scope="class")
def model_registry_instance_parametrized(
    request: FixtureRequest, admin_client: DynamicClient, teardown_resources: bool
) -> Generator[List[ModelRegistry], Any, Any]:
    """Create Model Registry instance parametrized"""
    with ExitStack() as stack:
        model_registry_instances = []
        mr_instances = [stack.enter_context(ModelRegistry(**param, client=admin_client)) for param in request.param]
        for mr_instance in mr_instances:
            # Common parameters for both ModelRegistry classes
            mr_instance.wait_for_condition(condition="Available", status="True")
            mr_instance.wait_for_condition(condition=KUBERBACPROXY_STR, status="True")
            model_registry_instances.append(mr_instance)

        LOGGER.info(
            f"Created {len(model_registry_instances)} MR instances: {[mr.name for mr in model_registry_instances]}"
        )
        yield model_registry_instances


@pytest.fixture()
def skip_test_on_byoidc(is_byoidc: bool) -> None:
    if is_byoidc:
        pytest.skip(reason="This test is meant to skip on byoidc.")
