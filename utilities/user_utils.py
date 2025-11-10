import logging
import shlex
import tempfile
from dataclasses import dataclass

from ocp_resources.user import User
from pyhelper_utils.shell import run_command
from timeout_sampler import retry

from utilities.exceptions import ExceptionUserLogin
from utilities.infra import login_with_user_password
import base64
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

LOGGER = logging.getLogger(__name__)
SLEEP_TIME = 5


@dataclass
class UserTestSession:
    """Represents a test user session with all necessary credentials and contexts."""

    __test__ = False
    idp_name: str
    secret_name: str
    username: str
    password: str
    original_user: str
    api_server_url: str
    is_byoidc: bool = False

    def __post_init__(self) -> None:
        """Validate the session data after initialization."""
        if self.is_byoidc:
            # we need to grab the username in byoidc mode
            with self.login():
                success, user, _ = run_command(command=["oc", "whoami"])
                if success:
                    self.username = user.strip()
                    LOGGER.info(f"Username in byoidc mode: {self.username}")
                else:
                    raise ValueError("Could not get username from oc whoami")
                return
        if not all([self.idp_name, self.secret_name, self.username, self.password]):
            raise ValueError("All session fields must be non-empty")
        if not (self.api_server_url and self.original_user):
            raise ValueError("Original user information and api url must be set")

    def cleanup(self) -> None:
        """Clean up the user context."""
        user = User(name=self.username)
        if user.exists:
            user.delete()

    @contextmanager
    def login(self) -> Generator[None, None, None]:
        if self.is_byoidc:
            current_context = run_command(command=["oc", "config", "current-context"])[1].strip()
            unprivileged_context = current_context + "-unprivileged"
            _ = run_command(command=["oc", "config", "use-context", unprivileged_context])
            yield
            _ = run_command(command=["oc", "config", "use-context", current_context])
        else:
            login_with_user_password(api_address=self.api_server_url, user=self.username, password=self.password)
            yield
            login_with_user_password(api_address=self.api_server_url, user=self.original_user)


def create_htpasswd_file(username: str, password: str) -> tuple[Path, str]:
    """
    Create an htpasswd file for a user.

    Args:
        username: The username to add to the htpasswd file
        password: The password for the user

    Returns:
        Tuple of (temp file path, base64 encoded content)
    """
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        temp_path = Path(temp_file.name).resolve()  # Get absolute path
        run_command(
            command=shlex.split(f"htpasswd -c -b {str(temp_path.absolute())} {username} {password}"), check=True
        )

        # Read the htpasswd file content and encode it
        temp_file.seek(0)  # noqa: FCN001 - TextIOWrapper.seek() doesn't accept keyword arguments
        htpasswd_content = temp_file.read()
        htpasswd_b64 = base64.b64encode(htpasswd_content.encode()).decode()

        return temp_path, htpasswd_b64


@retry(
    wait_timeout=240,
    sleep=10,
    exceptions_dict={ExceptionUserLogin: []},
)
def wait_for_user_creation(username: str, password: str, cluster_url: str) -> bool:
    """
    Attempts to login to OpenShift as a specific user over a period of time to ensure user creation

    Args:
        username: The username to login with
        password: The password to login with
        cluster_url: The OpenShift cluster URL

    Returns:
        True if login is successful
    """
    # not executed in byoidc mode
    LOGGER.info(f"Attempting to login as {username}")
    res = login_with_user_password(api_address=cluster_url, user=username, password=password)

    if res:
        return True
    raise ExceptionUserLogin(f"Could not login as user {username}.")
