from contextlib import ExitStack

import pytest
from typing import Generator, List, Any

from _pytest.fixtures import FixtureRequest
from simple_logger.logger import get_logger

from ocp_resources.deployment import Deployment
from ocp_resources.infrastructure import Infrastructure
from ocp_resources.model_registry_modelregistry_opendatahub_io import ModelRegistry
from ocp_resources.oauth import OAuth
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.role_binding import RoleBinding
from ocp_resources.role import Role
from ocp_resources.group import Group

from ocp_resources.resource import ResourceEditor
from kubernetes.dynamic import DynamicClient
from pyhelper_utils.shell import run_command

from tests.model_registry.rbac.utils import wait_for_oauth_openshift_deployment, create_role_binding
from tests.model_registry.utils import delete_model_catalog_configmap
from utilities.general import generate_random_name
from utilities.infra import login_with_user_password
from utilities.user_utils import UserTestSession, create_htpasswd_file, wait_for_user_creation
from tests.model_registry.rbac.group_utils import create_group
from tests.model_registry.constants import (
    MR_INSTANCE_NAME,
)
from pytest_testconfig import config as py_config

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


@pytest.fixture(scope="module")
def user_credentials_rbac() -> dict[str, str]:
    random_str = generate_random_name()
    return {
        "username": f"test-user-{random_str}",
        "password": f"test-password-{random_str}",
        "idp_name": f"test-htpasswd-idp-{random_str}",
        "secret_name": f"test-htpasswd-secret-{random_str}",
    }


@pytest.fixture(scope="session")
def original_user() -> str:
    current_user = run_command(command=["oc", "whoami"])[1].strip()
    LOGGER.info(f"Original user: {current_user}")
    return current_user


@pytest.fixture(scope="module")
def created_htpasswd_secret(
    original_user: str, user_credentials_rbac: dict[str, str]
) -> Generator[UserTestSession, None, None]:
    """
    Session-scoped fixture that creates a test IDP user and cleans it up after all tests.
    Returns a UserTestSession object that contains all necessary credentials and contexts.
    """

    temp_path, htpasswd_b64 = create_htpasswd_file(
        username=user_credentials_rbac["username"], password=user_credentials_rbac["password"]
    )
    try:
        LOGGER.info(f"Creating secret {user_credentials_rbac['secret_name']} in openshift-config namespace")
        with Secret(
            name=user_credentials_rbac["secret_name"],
            namespace="openshift-config",
            htpasswd=htpasswd_b64,
            type="Opaque",
            wait_for_resource=True,
        ) as secret:
            yield secret
    finally:
        # Clean up the temporary file
        temp_path.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def updated_oauth_config(
    admin_client: DynamicClient, original_user: str, user_credentials_rbac: dict[str, str]
) -> Generator[Any, None, None]:
    # Get current providers and add the new one
    oauth = OAuth(name="cluster")
    identity_providers = oauth.instance.spec.identityProviders

    new_idp = {
        "name": user_credentials_rbac["idp_name"],
        "mappingMethod": "claim",
        "type": "HTPasswd",
        "htpasswd": {"fileData": {"name": user_credentials_rbac["secret_name"]}},
    }
    updated_providers = identity_providers + [new_idp]

    LOGGER.info("Updating OAuth")
    identity_providers_patch = ResourceEditor(patches={oauth: {"spec": {"identityProviders": updated_providers}}})
    identity_providers_patch.update(backup_resources=True)
    # Wait for OAuth server to be ready
    wait_for_oauth_openshift_deployment()
    LOGGER.info(f"Added IDP {user_credentials_rbac['idp_name']} to OAuth configuration")
    yield
    identity_providers_patch.restore()
    wait_for_oauth_openshift_deployment()


@pytest.fixture(scope="module")
def test_idp_user(
    original_user: str,
    user_credentials_rbac: dict[str, str],
    created_htpasswd_secret: Generator[UserTestSession, None, None],
    updated_oauth_config: Generator[Any, None, None],
    api_server_url: str,
) -> Generator[UserTestSession, None, None]:
    """
    Session-scoped fixture that creates a test IDP user and cleans it up after all tests.
    Returns a UserTestSession object that contains all necessary credentials and contexts.
    """
    idp_session = None
    try:
        if wait_for_user_creation(
            username=user_credentials_rbac["username"],
            password=user_credentials_rbac["password"],
            cluster_url=api_server_url,
        ):
            # undo the login as test user if we were successful in logging in as test user
            LOGGER.info(f"Undoing login as test user and logging in as {original_user}")
            login_with_user_password(api_address=api_server_url, user=original_user)

        idp_session = UserTestSession(
            idp_name=user_credentials_rbac["idp_name"],
            secret_name=user_credentials_rbac["secret_name"],
            username=user_credentials_rbac["username"],
            password=user_credentials_rbac["password"],
            original_user=original_user,
            api_server_url=api_server_url,
        )
        LOGGER.info(f"Created session test IDP user: {idp_session.username}")

        yield idp_session

    finally:
        if idp_session:
            LOGGER.info(f"Cleaning up test IDP user: {idp_session.username}")
            idp_session.cleanup()


@pytest.fixture()
def login_as_test_user(
    api_server_url: str, original_user: str, test_idp_user: UserTestSession
) -> Generator[None, None, None]:
    LOGGER.info(f"Logging in as {test_idp_user.username}")
    login_with_user_password(
        api_address=api_server_url,
        user=test_idp_user.username,
        password=test_idp_user.password,
    )
    yield
    LOGGER.info(f"Logging in as {original_user}")
    login_with_user_password(
        api_address=api_server_url,
        user=original_user,
    )


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
    admin_client: DynamicClient,
    model_registry_namespace: str,
    mr_access_role: Role,
    test_idp_user: UserTestSession,
) -> Generator[RoleBinding, None, None]:
    yield from create_role_binding(
        admin_client=admin_client,
        model_registry_namespace=model_registry_namespace,
        name="test-model-registry-access",
        mr_access_role=mr_access_role,
        subjects_kind="User",
        subjects_name=test_idp_user.username,
    )


# =============================================================================
# RESOURCE FIXTURES PARMETRIZED
# =============================================================================
@pytest.fixture(scope="class")
def db_secret_parametrized(request: FixtureRequest, teardown_resources: bool) -> Generator[List[Secret], Any, Any]:
    """Create DB Secret parametrized"""
    with ExitStack() as stack:
        secrets = [
            stack.enter_context(
                Secret(
                    **param,
                    teardown=teardown_resources,
                )
            )
            for param in request.param
        ]
        yield secrets


@pytest.fixture(scope="class")
def db_pvc_parametrized(
    request: FixtureRequest, teardown_resources: bool
) -> Generator[List[PersistentVolumeClaim], Any, Any]:
    """Create DB PVC parametrized"""
    with ExitStack() as stack:
        pvc = [
            stack.enter_context(
                PersistentVolumeClaim(
                    **param,
                    teardown=teardown_resources,
                )
            )
            for param in request.param
        ]
        yield pvc


@pytest.fixture(scope="class")
def db_service_parametrized(request: FixtureRequest, teardown_resources: bool) -> Generator[List[Service], Any, Any]:
    """Create DB Service parametrized"""
    with ExitStack() as stack:
        services = [
            stack.enter_context(
                Service(
                    **param,
                    teardown=teardown_resources,
                )
            )
            for param in request.param
        ]
        yield services


@pytest.fixture(scope="class")
def db_deployment_parametrized(
    request: FixtureRequest, teardown_resources: bool
) -> Generator[List[Deployment], Any, Any]:
    """Create DB Deployment parametrized"""
    with ExitStack() as stack:
        deployments = [
            stack.enter_context(
                Deployment(
                    **param,
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
        for param in request.param:
            # Common parameters for both ModelRegistry classes
            mr_instance = stack.enter_context(ModelRegistry(**param))  # noqa: FCN001
            mr_instance.wait_for_condition(condition="Available", status="True")
            mr_instance.wait_for_condition(condition="OAuthProxyAvailable", status="True")
            model_registry_instances.append(mr_instance)

        LOGGER.info(
            f"Created {len(model_registry_instances)} MR instances: {[mr.name for mr in model_registry_instances]}"
        )
        yield model_registry_instances
    # delete the model catalog configmap manually:
    delete_model_catalog_configmap(admin_client=admin_client, namespace=py_config["model_registry_namespace"])


@pytest.fixture(scope="session")
def api_server_url(admin_client: DynamicClient) -> str:
    """
    Get api server url from the cluster.
    """
    infrastructure = Infrastructure(client=admin_client, name="cluster", ensure_exists=True)
    return infrastructure.instance.status.apiServerURL
