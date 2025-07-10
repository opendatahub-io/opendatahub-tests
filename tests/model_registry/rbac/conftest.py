import pytest
import shlex
import subprocess
import os
from typing import Generator, List, Dict, Any

from simple_logger.logger import get_logger

from ocp_resources.namespace import Namespace
from ocp_resources.oauth import OAuth
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.role_binding import RoleBinding
from ocp_resources.role import Role
from ocp_resources.group import Group
from ocp_resources.resource import ResourceEditor
from kubernetes.dynamic import DynamicClient
from pyhelper_utils.shell import run_command

from tests.model_registry.rbac.utils import wait_for_oauth_openshift_deployment, create_role_binding
from utilities.general import generate_random_name
from tests.model_registry.utils import generate_namespace_name
from utilities.infra import login_with_user_password
from utilities.user_utils import UserTestSession, create_htpasswd_file, wait_for_user_creation
from tests.model_registry.rbac.group_utils import create_group
from tests.model_registry.constants import MR_INSTANCE_NAME


LOGGER = get_logger(name=__name__)
DEFAULT_TOKEN_DURATION = "10m"


@pytest.fixture(scope="function")
def sa_namespace(request: pytest.FixtureRequest, admin_client: DynamicClient) -> Generator[Namespace, None, None]:
    """
    Creates a temporary namespace using a context manager for automatic cleanup.
    Function scope ensures a fresh namespace for each test needing it.
    """
    test_file = os.path.relpath(request.fspath.strpath, start=os.path.dirname(__file__))
    ns_name = generate_namespace_name(file_path=test_file)
    LOGGER.info(f"Creating temporary namespace: {ns_name}")
    with Namespace(client=admin_client, name=ns_name) as ns:
        ns.wait_for_status(status=Namespace.Status.ACTIVE, timeout=120)
        yield ns


@pytest.fixture(scope="function")
def service_account(admin_client: DynamicClient, sa_namespace: Namespace) -> Generator[ServiceAccount, None, None]:
    """
    Creates a ServiceAccount within the temporary namespace using a context manager.
    Function scope ensures it's tied to the lifetime of sa_namespace for that test.
    """
    sa_name = generate_random_name(prefix="mr-test-user")
    LOGGER.info(f"Creating ServiceAccount: {sa_name} in namespace {sa_namespace.name}")
    with ServiceAccount(client=admin_client, name=sa_name, namespace=sa_namespace.name, wait_for_resource=True) as sa:
        yield sa


@pytest.fixture(scope="function")
def sa_token(service_account: ServiceAccount) -> str:
    """
    Retrieves a short-lived token for the ServiceAccount using 'oc create token'.
    Function scope because token is temporary and tied to the SA for that test.
    """
    sa_name = service_account.name
    namespace = service_account.namespace
    LOGGER.info(f"Retrieving token for ServiceAccount: {sa_name} in namespace {namespace}")
    try:
        cmd = f"oc create token {sa_name} -n {namespace} --duration={DEFAULT_TOKEN_DURATION}"
        LOGGER.debug(f"Executing command: {cmd}")
        res, out, err = run_command(command=shlex.split(cmd), verify_stderr=False, check=True, timeout=30)
        token = out.strip()
        if not token:
            raise ValueError("Retrieved token is empty after successful command execution.")

        LOGGER.info(f"Successfully retrieved token for SA '{sa_name}'")
        return token

    except Exception as e:  # Catch all exceptions from the try block
        error_type = type(e).__name__
        log_message = (
            f"Failed during token retrieval for SA '{sa_name}' in namespace '{namespace}'. "
            f"Error Type: {error_type}, Message: {str(e)}"
        )
        if isinstance(e, subprocess.CalledProcessError):
            # Add specific details for CalledProcessError
            # run_command already logs the error if log_errors=True and returncode !=0,
            # but we can add context here.
            stderr_from_exception = e.stderr.strip() if e.stderr else "N/A"
            log_message += f". Exit Code: {e.returncode}. Stderr from exception: {stderr_from_exception}"
        elif isinstance(e, subprocess.TimeoutExpired):
            timeout_value = getattr(e, "timeout", "N/A")
            log_message += f". Command timed out after {timeout_value} seconds."
        elif isinstance(e, FileNotFoundError):
            # This occurs if 'oc' is not found.
            # e.filename usually holds the name of the file that was not found.
            command_not_found = e.filename if hasattr(e, "filename") and e.filename else shlex.split(cmd)[0]
            log_message += f". Command '{command_not_found}' not found. Is it installed and in PATH?"

        LOGGER.error(log_message, exc_info=True)  # exc_info=True adds stack trace to the log
        raise


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


@pytest.fixture(scope="function")
def mr_access_role(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    sa_namespace: Namespace,
) -> Generator[Role, None, None]:
    """
    Creates the MR Access Role using direct constructor parameters and a context manager.
    """
    role_name = f"registry-user-{MR_INSTANCE_NAME}-{sa_namespace.name[:8]}"
    LOGGER.info(f"Defining Role: {role_name} in namespace {model_registry_namespace}")

    role_rules: List[Dict[str, Any]] = [
        {
            "apiGroups": [""],
            "resources": ["services"],
            "resourceNames": [MR_INSTANCE_NAME],  # Grant access only to the specific MR service object
            "verbs": ["get"],
        }
    ]
    role_labels = {
        "app.kubernetes.io/component": "model-registry-test-rbac",
        "test.opendatahub.io/namespace": sa_namespace.name,
    }

    LOGGER.info(f"Attempting to create Role: {role_name} with rules and labels.")
    with Role(
        client=admin_client,
        name=role_name,
        namespace=model_registry_namespace,
        rules=role_rules,
        label=role_labels,
        wait_for_resource=True,
    ) as role:
        LOGGER.info(f"Role {role.name} created successfully.")
        yield role


@pytest.fixture(scope="function")
def mr_access_role_binding(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    mr_access_role: Role,
    sa_namespace: Namespace,
) -> Generator[RoleBinding, None, None]:
    """
    Creates the MR Access RoleBinding using direct constructor parameters and a context manager.
    """
    binding_name = f"{mr_access_role.name}-binding"

    LOGGER.info(
        f"Defining RoleBinding: {binding_name} linking Group 'system:serviceaccounts:{sa_namespace.name}' "
        f"to Role '{mr_access_role.name}' in namespace {model_registry_namespace}"
    )
    binding_labels = {
        "app.kubernetes.io/component": "model-registry-test-rbac",
        "test.opendatahub.io/namespace": sa_namespace.name,
    }

    LOGGER.info(f"Attempting to create RoleBinding: {binding_name} with labels.")
    with RoleBinding(
        client=admin_client,
        name=binding_name,
        namespace=model_registry_namespace,
        # Subject parameters
        subjects_kind="Group",
        subjects_name=f"system:serviceaccounts:{sa_namespace.name}",
        subjects_api_group="rbac.authorization.k8s.io",  # This is the default apiGroup for Group kind
        # Role reference parameters
        role_ref_kind=mr_access_role.kind,
        role_ref_name=mr_access_role.name,
        label=binding_labels,
        wait_for_resource=True,
    ) as binding:
        LOGGER.info(f"RoleBinding {binding.name} created successfully.")
        yield binding
        LOGGER.info(f"RoleBinding {binding.name} deletion initiated by context manager.")


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
