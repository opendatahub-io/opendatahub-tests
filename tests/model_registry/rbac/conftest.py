import pytest
import os
from typing import Callable, ContextManager, Generator
from contextlib import contextmanager
from pyhelper_utils.shell import run_command

from tests.model_registry.constants import MR_INSTANCE_NAME
from kubernetes.dynamic import DynamicClient
from ocp_resources.role import Role
from simple_logger.logger import get_logger

LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="session", autouse=True)
def check_env_vars() -> None:
    """
    Check if the required environment variables are set.
    Fails the test session if any are missing.
    """
    missing: list[str] = []
    missing.extend(var for var in ["ADMIN_PASSWORD", "NON_ADMIN_PASSWORD"] if not os.environ.get(var))
    if missing:
        pytest.skip(f"Required environment variables not set: {', '.join(missing)}")


@pytest.fixture(scope="function")
def user_in_group_context() -> Callable[[str], ContextManager[None]]:
    """
    Fixture to add and remove a user from the model-registry-users group.
    Returns a context manager that adds the user on entry and removes on exit.
    """

    @contextmanager
    def _context(user_name: str) -> Generator[None, None, None]:
        run_command(command=["oc", "adm", "groups", "add-users", "model-registry-users", user_name])
        try:
            yield
        finally:
            run_command(command=["oc", "adm", "groups", "remove-users", "model-registry-users", user_name])

    return _context


@pytest.fixture
def new_group(request: pytest.FixtureRequest) -> Generator[str, None, None]:
    """
    Fixture to create a new OpenShift group and add a user, then delete the group after the test.
    The parameter should be a tuple: (group_name, user_name).
    """
    group_name, user_name = request.param
    run_command(command=["oc", "adm", "groups", "new", group_name, user_name])
    try:
        yield group_name
    finally:
        run_command(command=["oc", "delete", "group", group_name])


@pytest.fixture(scope="function")
def model_registry_role(
    admin_client: DynamicClient,
    model_registry_namespace: str,
) -> Generator[Role, None, None]:
    """
    Fixture to create a Role for Model Registry access.
    """
    with Role(
        client=admin_client,
        namespace=model_registry_namespace,
        name="test-model-registry-get",
        rules=[
            {
                "apiGroups": [""],
                "resources": ["services"],
                "resourceNames": [MR_INSTANCE_NAME],
                "verbs": ["get"],
            },
        ],
        label={
            "app.kubernetes.io/component": "model-registry-test-rbac",
            "test.opendatahub.io/namespace": model_registry_namespace,
        },
        wait_for_resource=True,
    ) as role:
        try:
            LOGGER.info(f"Role {role.name} created successfully.")
            yield role
            LOGGER.info(f"Role {role.name} deleted successfully.")
        except Exception as e:  # Catch other potential errors during Role instantiation or wait
            LOGGER.error(f"Error during Role {role.name} creation or wait: {e}", exc_info=True)
            pytest.fail(f"Failed during Role {role.name} creation: {e}")
