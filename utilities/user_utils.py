import logging
import tempfile
import subprocess
from dataclasses import dataclass
from pyhelper_utils.shell import run_command
from timeout_sampler import TimeoutSampler
from utilities.infra import login_with_user_password, get_client, switch_user_context
from tests.model_registry.utils import generate_random_name, wait_for_pods_running
from contextlib import contextmanager
from typing import Generator, Optional
from ocp_resources.oauth import OAuth
from ocp_resources.resource import ResourceEditor
from ocp_resources.secret import Secret
import base64
from pathlib import Path


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
    user_context: str
    original_context: str

    def __post_init__(self) -> None:
        """Validate the session data after initialization."""
        if not all([self.idp_name, self.secret_name, self.username, self.password]):
            raise ValueError("All session fields must be non-empty")
        if not self.user_context or not self.original_context:
            raise ValueError("Both user and original contexts must be set")

    def cleanup(self) -> None:
        """Clean up the user context."""
        if self.user_context:
            # First switch to original context
            run_command(command=["oc", "config", "use-context", self.original_context], check=True)
            LOGGER.info(f"Switched to original context: {self.original_context}")
            # Then delete the user context
            LOGGER.info(f"Deleting user context: {self.user_context}")
            run_command(
                command=["oc", "config", "delete-context", self.user_context],
                check=False,  # Don't fail if context doesn't exist
            )


@contextmanager
def create_test_idp(
    idp_name: str = "test-htpasswd-idp", secret_name: str = "test-htpasswd-secret"
) -> Generator[UserTestSession, None, None]:
    """
    Context manager to create and manage a test HTPasswd IDP in OpenShift.
    Creates the IDP and test user, then cleans up after use.

    Args:
        idp_name: Name for the IDP
        secret_name: Name for the secret

    Yields:
        UserTestSession object containing user credentials and contexts

    Example:
        with create_test_idp() as idp_session:
            # Use idp_session here
            # Cleanup happens automatically after the with block
    """
    # Save the current context (cluster admin)
    _, original_context, _ = run_command(command=["oc", "config", "current-context"], check=True)
    original_context = original_context.strip()
    LOGGER.info(f"Original context (cluster admin): {original_context}")

    # Generate unique names
    idp_name = generate_random_name(prefix=idp_name)
    secret_name = generate_random_name(prefix=secret_name)
    username = generate_random_name(prefix="test-user")
    password = generate_random_name(prefix="test-password")

    idp_session = None
    try:
        LOGGER.info(f"Creating user with username: {username}")

        with _create_htpasswd_secret(username=username, password=password, secret_name=secret_name):
            with _update_oauth_config(idp_name=idp_name, secret_name=secret_name):
                # Get the cluster URL
                _, cluster_url, _ = run_command(command=["oc", "whoami", "--show-server"], check=True)
                cluster_url = cluster_url.strip()

                # Use TimeoutSampler to retry user context creation
                sampler = TimeoutSampler(
                    wait_timeout=240,
                    sleep=10,
                    func=_create_user_context,
                    username=username,
                    password=password,
                    cluster_url=cluster_url,
                )

                user_context = None
                for user_context in sampler:
                    if user_context:  # If context creation was successful
                        break

                if not user_context:
                    raise Exception(f"Could not create context for user {username} after timeout")

                # Switch back to original context after creating user context
                with switch_user_context(original_context):
                    idp_session = UserTestSession(
                        idp_name=idp_name,
                        secret_name=secret_name,
                        username=username,
                        password=password,
                        user_context=user_context,
                        original_context=original_context,
                    )
                    yield idp_session

    except Exception as e:
        LOGGER.error(f"Error during setup: {e}")
        raise
    finally:
        if idp_session:
            LOGGER.info(f"Cleaning up test IDP user: {idp_session.username}")
            idp_session.cleanup()


def _create_htpasswd_file(username: str, password: str) -> tuple[Path, str]:
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
        subprocess.run(args=["htpasswd", "-c", "-b", str(temp_path.absolute()), username, password], check=True)

        # Read the htpasswd file content and encode it
        temp_file.seek(0)  # noqa: FCN001 - TextIOWrapper.seek() doesn't accept keyword arguments
        htpasswd_content = temp_file.read()
        htpasswd_b64 = base64.b64encode(htpasswd_content.encode()).decode()

        return temp_path, htpasswd_b64


@contextmanager
def _create_htpasswd_secret(username: str, password: str, secret_name: str) -> Generator[Secret, None, None]:
    """
    Creates an htpasswd file and corresponding OpenShift secret for a user.
    Uses context manager to ensure proper cleanup.

    Args:
        username: The username to add to the htpasswd file
        password: The password for the user
        secret_name: The name of the secret to create

    Yields:
        Secret: The created Secret resource
    """
    temp_path, htpasswd_b64 = _create_htpasswd_file(username=username, password=password)
    try:
        LOGGER.info(f"Creating secret {secret_name} in openshift-config namespace")
        with Secret(
            name=secret_name,
            namespace="openshift-config",
            htpasswd=htpasswd_b64,
            type="Opaque",
            wait_for_resource=True,
        ) as secret:
            yield secret
    finally:
        # Clean up the temporary file
        temp_path.unlink(missing_ok=True)


@contextmanager
def _update_oauth_config(idp_name: str, secret_name: str) -> Generator[None, None, None]:
    """
    Updates the OpenShift OAuth configuration to add a new HTPasswd identity provider.
    Uses ResourceEditor context manager to handle cleanup.

    Args:
        idp_name: The name of the identity provider to add
        secret_name: The name of the secret containing the htpasswd file

    Yields:
        None, but ensures cleanup through ResourceEditor context manager
    """
    # Get current providers and add the new one
    oauth = OAuth(name="cluster")
    current_oauth = oauth.instance
    identity_providers = current_oauth.spec.identityProviders

    new_idp = {
        "name": idp_name,
        "mappingMethod": "claim",
        "type": "HTPasswd",
        "htpasswd": {"fileData": {"name": secret_name}},
    }
    updated_providers = identity_providers + [new_idp]

    LOGGER.info("Updating OAuth")
    with ResourceEditor(patches={oauth: {"spec": {"identityProviders": updated_providers}}}) as _:
        # Wait for OAuth server to be ready
        wait_for_pods_running(
            admin_client=get_client(), namespace_name="openshift-authentication", number_of_consecutive_checks=1
        )
        LOGGER.info(f"Added IDP {idp_name} to OAuth configuration")
        yield


def _create_user_context(username: str, password: str, cluster_url: str) -> Optional[str]:
    """
    Attempts to login to OpenShift to create a user context.

    Args:
        username: The username to login with
        password: The password to login with
        cluster_url: The OpenShift cluster URL

    Returns:
        The user context name if login successful, None otherwise

    Note:
        This function is meant to be used with TimeoutSampler for retries.
        It will return None on failure, allowing the sampler to retry.
    """
    LOGGER.info(f"Attempting to login as {username}")

    if login_with_user_password(api_address=cluster_url, user=username, password=password):
        # Login was successful, get the current context
        _, stdout, _ = run_command(command=["oc", "config", "current-context"], check=True)
        user_context = stdout.strip()
        LOGGER.info(f"Successfully created context for user {username}: {user_context}")
        return user_context

    LOGGER.error(f"Login failed for user {username}")
    return None
