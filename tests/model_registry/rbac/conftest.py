import pytest
import os
from typing import Callable, ContextManager, Generator
from contextlib import contextmanager
from pyhelper_utils.shell import run_command


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


@pytest.fixture
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
