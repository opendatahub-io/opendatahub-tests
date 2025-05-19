import pytest
import shlex
import subprocess
import os
from typing import Generator, List, Dict, Any
from ocp_resources.namespace import Namespace
from ocp_resources.service_account import ServiceAccount
from ocp_resources.role_binding import RoleBinding
from ocp_resources.role import Role
from kubernetes.dynamic import DynamicClient
from pyhelper_utils.shell import run_command
from tests.model_registry.utils import generate_random_name, generate_namespace_name
from simple_logger.logger import get_logger
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


# --- RBAC Fixtures ---


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
        LOGGER.info(f"Role {role.name} deletion initiated by context manager.")


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
        role_ref_kind="Role",
        role_ref_name=mr_access_role.name,
        label=binding_labels,
        wait_for_resource=True,
    ) as binding:
        LOGGER.info(f"RoleBinding {binding.name} created successfully.")
        yield binding
        LOGGER.info(f"RoleBinding {binding.name} deletion initiated by context manager.")
