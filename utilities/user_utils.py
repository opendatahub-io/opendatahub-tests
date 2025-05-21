import logging
import tempfile
import subprocess
from dataclasses import dataclass
from pyhelper_utils.shell import run_command
from timeout_sampler import TimeoutSampler
from utilities.infra import login_with_user_password, get_client
from tests.model_registry.utils import generate_random_name, wait_for_pods_running
from contextlib import contextmanager
from typing import Generator
from ocp_resources.oauth import OAuth
from ocp_resources.resource import ResourceEditor


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
    user_context = ""
    idp_session = None
    # Save the current context (cluster admin)
    _, original_context, _ = run_command(command=["oc", "config", "current-context"], check=True)
    original_context = original_context.strip()
    LOGGER.info(f"Original context (cluster admin): {original_context}")
    # Generate a unique suffix for this test run
    idp_name = generate_random_name(prefix=idp_name)
    secret_name = generate_random_name(prefix=secret_name)

    # Generate username and password
    username = generate_random_name(prefix="test-user")
    password = generate_random_name(prefix="test-password")

    try:
        LOGGER.info(f"Creating user with username: {username}")

        _create_htpasswd_secret(username=username, password=password, secret_name=secret_name)
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

            for user_context in sampler:
                if user_context:  # If context creation was successful
                    break

            if not user_context:
                raise Exception(f"Could not create context for user {username} after timeout")

            # Switch back to original context after creating user context
            run_command(command=["oc", "config", "use-context", original_context], check=True)
            LOGGER.info(f"Switched back to original context: {original_context}")

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
            cleanup_test_idp(
                idp_name=idp_session.idp_name,
                secret_name=idp_session.secret_name,
                original_context=original_context,
                user_context=user_context,
            )


def cleanup_test_idp(idp_name: str, secret_name: str, original_context: str, user_context: str) -> None:
    """
    Cleans up test IDP resources and restores the original context.
    Note: The OAuth configuration is automatically restored by ResourceEditor's context manager.

    Args:
        idp_name: Name of the IDP to remove (not used, kept for compatibility)
        secret_name: Name of the secret to delete
        original_context: Name of the original context (cluster admin) to restore
        user_context: Name of the user context to delete
    """
    cleanup_errors = []

    try:
        # Restore the original context
        _restore_context_and_delete_user_context(original_context=original_context, user_context=user_context)

        # Delete the secret
        LOGGER.info(f"Deleting secret {secret_name}...")
        try:
            run_command(
                command=["oc", "delete", "secret", secret_name, "-n", "openshift-config"],
                check=False,  # Don't fail if secret doesn't exist
            )
        except Exception as e:
            cleanup_errors.append(f"Failed to delete secret {secret_name}: {str(e)}")
            LOGGER.error(f"Error deleting secret: {e}")

        if cleanup_errors:
            LOGGER.warning("Cleanup completed with some errors:")
            for error in cleanup_errors:
                LOGGER.warning(f"- {error}")
        else:
            LOGGER.info("Cleanup completed successfully")

    except Exception as e:
        LOGGER.error(f"Critical error during cleanup: {e}")
        # Try to restore the original context even if cleanup fails
        if original_context:
            try:
                run_command(command=["oc", "config", "use-context", original_context], check=True)
                LOGGER.info(f"Restored original context after error: {original_context}")
            except Exception as restore_error:
                LOGGER.error(f"Failed to restore original context: {restore_error}")
        raise

    # If we had any cleanup errors, raise them now
    if cleanup_errors:
        raise Exception("Cleanup completed with errors:\n" + "\n".join(cleanup_errors))


def _create_user_context(
    username: str,
    password: str,
    cluster_url: str,
) -> str | None:
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


def _create_htpasswd_secret(username: str, password: str, secret_name: str) -> None:
    """
    Creates an htpasswd file and corresponding OpenShift secret for a user.

    This function generates an htpasswd file for the given username and password,
    then creates a Kubernetes secret in the 'openshift-config' namespace using that file.

    Args:
        username: The username to add to the htpasswd file.
        password: The password for the user.
        secret_name: The name of the secret to create.
    """
    with tempfile.NamedTemporaryFile(mode="w+") as temp_file:
        subprocess.run(args=["htpasswd", "-c", "-b", temp_file.name, username, password], check=True)
        LOGGER.info(f"Creating secret {secret_name} in openshift-config namespace")
        run_command(
            command=[
                "oc",
                "create",
                "secret",
                "generic",
                secret_name,
                f"--from-file=htpasswd={temp_file.name}",
                "-n",
                "openshift-config",
            ],
            check=True,
        )

        # Verify the secret was created correctly
        _, secret_data, _ = run_command(
            command=["oc", "get", "secret", secret_name, "-n", "openshift-config", "-o", "jsonpath={.data.htpasswd}"],
            check=True,
        )
        if not secret_data:
            raise Exception(f"Secret {secret_name} was created but has no htpasswd data")

        LOGGER.info(f"Created secret: {secret_name}")


@contextmanager
def _update_oauth_config(idp_name: str, secret_name: str) -> Generator[None, None, None]:
    """
    Updates the OpenShift OAuth configuration to add a new HTPasswd identity provider.
    Uses ResourceEditor context manager to handle cleanup.

    Args:
        idp_name: The name of the identity provider to add.
        secret_name: The name of the secret containing the htpasswd file.

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


def _restore_context_and_delete_user_context(original_context: str, user_context: str) -> None:
    """
    Restores the original OpenShift context and deletes the specified user context.

    Args:
        original_context: The name of the context to restore.
        user_context: The name of the user context to delete.
    """
    LOGGER.info(f"Restoring original context: {original_context}")
    run_command(command=["oc", "config", "use-context", original_context], check=True)
    if user_context:
        # Delete the user context if created
        LOGGER.info(f"Deleting user context: {user_context}")
        run_command(
            command=["oc", "config", "delete-context", user_context],
            check=False,  # Don't fail if context doesn't exist
        )
