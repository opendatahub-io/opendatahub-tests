import base64
import logging
import shlex
import tempfile
from dataclasses import dataclass
from pathlib import Path

import requests
from kubernetes.dynamic import DynamicClient
from ocp_resources.user import User
from pyhelper_utils.shell import run_command
from timeout_sampler import retry

from utilities.exceptions import ExceptionUserLogin
from utilities.infra import get_cluster_authentication, login_with_user_password

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
    client: DynamicClient
    is_byoidc: bool = False

    def __post_init__(self) -> None:
        """Validate the session data after initialization."""
        if not all([self.idp_name, self.secret_name, self.username, self.password]):
            raise ValueError("All session fields must be non-empty")
        if not (self.api_server_url and self.original_user):
            raise ValueError("Original user information and api url must be set")
        if self.client is None:
            raise ValueError("Client must be provided")

    def cleanup(self) -> None:
        """Clean up the user context."""
        user = User(name=self.username, client=self.client)
        if user.exists:
            user.delete()


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
        run_command(command=shlex.split(f"htpasswd -c -b {temp_path.absolute()!s} {username} {password}"), check=True)

        # Read the htpasswd file content and encode it
        temp_file.seek(0)
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


def get_oidc_tokens(admin_client: DynamicClient, username: str, password: str) -> tuple[str, str]:
    url = f"{get_byoidc_issuer_url(admin_client=admin_client)}/protocol/openid-connect/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded", "User-Agent": "python-requests"}

    data = {
        "username": username,
        "password": password,
        "grant_type": "password",
        "client_id": "oc-cli",
        "scope": "openid",
    }

    LOGGER.info(f"Requesting token for user {username} in byoidc environment")
    response = requests.post(
        url=url,
        headers=headers,
        data=data,
        allow_redirects=True,
        timeout=30,
        verify=True,  # Set to False if you need to skip SSL verification
    )
    response.raise_for_status()
    json_response = response.json()

    # Validate that we got an access token
    if "id_token" not in json_response or "refresh_token" not in json_response:
        LOGGER.error("Warning: No id_token or refresh_token in response")
        raise AssertionError(f"No id_token or refresh_token in response: {json_response}")
    return json_response["id_token"], json_response["refresh_token"]


def get_byoidc_issuer_url(admin_client: DynamicClient) -> str:
    authentication = get_cluster_authentication(admin_client=admin_client)
    assert authentication is not None
    url = authentication.instance.spec.oidcProviders[0].issuer.issuerURL
    assert url is not None
    return url
