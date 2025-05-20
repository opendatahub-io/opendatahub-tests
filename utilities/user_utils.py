import secrets
import string
import random
import logging
import tempfile
import subprocess
import os
import json
from dataclasses import dataclass
from pyhelper_utils.shell import run_command
from timeout_sampler import TimeoutSampler
from utilities.infra import login_with_user_password


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


def create_test_idp(idp_name: str = "test-htpasswd-idp", secret_name: str = "test-htpasswd-secret") -> UserTestSession:
    """
    Creates a new HTPasswd IDP in OpenShift and adds a test user.

    Args:
        idp_name: Name for the IDP
        secret_name: Name for the secret

    Returns:
        UserTestSession object
    """
    user_context = ""
    try:
        # Save the current context (cluster admin)
        _, original_context, _ = run_command(command=["oc", "config", "current-context"], check=True)
        original_context = original_context.strip()
        LOGGER.info(f"Original context (cluster admin): {original_context}")

        # Generate a unique suffix for this test run
        unique_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
        idp_name = f"{idp_name}-{unique_suffix}"
        secret_name = f"{secret_name}-{unique_suffix}"

        # Generate username and password
        username = f"test-user-{secrets.token_hex(4)}"
        password = "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(12))

        LOGGER.info(f"Creating user with username: {username}")

        _create_htpasswd_secret(username=username, password=password, secret_name=secret_name)
        _update_oauth_config(idp_name=idp_name, secret_name=secret_name)

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

        return UserTestSession(
            idp_name=idp_name,
            secret_name=secret_name,
            username=username,
            password=password,
            user_context=user_context,
            original_context=original_context,
        )

    except Exception as e:
        LOGGER.error(f"Error during setup: {e}")
        cleanup_test_idp(idp_name, secret_name, original_context, user_context)
        raise


def cleanup_test_idp(idp_name: str, secret_name: str, original_context: str, user_context: str) -> None:
    """
    Cleans up test IDP resources and restores the original context.

    Args:
        idp_name: Name of the IDP to remove
        secret_name: Name of the secret to delete
        original_context: Name of the original context (cluster admin) to restore
        user_context: Name of the user context to delete
    """
    try:
        # Restore the original context
        _restore_context_and_delete_user_context(original_context=original_context, user_context=user_context)

        _remove_idp_from_oauth(idp_name=idp_name)

        # Delete the secret
        LOGGER.info(f"Deleting secret: {secret_name}")
        run_command(
            command=["oc", "delete", "secret", secret_name, "-n", "openshift-config"],
            check=False,  # Don't fail if secret doesn't exist
        )

        LOGGER.info("Cleanup completed successfully")

    except Exception as e:
        LOGGER.error(f"Error during cleanup: {e}")
        # Try to restore the original context even if cleanup fails
        if original_context:
            try:
                run_command(command=["oc", "config", "use-context", original_context], check=True)
                LOGGER.info(f"Restored original context after error: {original_context}")
            except Exception as restore_error:
                LOGGER.error(f"Failed to restore original context: {restore_error}")
        raise


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
        LOGGER.info(f"Created secret: {secret_name}")


def _update_oauth_config(idp_name: str, secret_name: str) -> None:
    """
    Updates the OpenShift OAuth configuration to add a new HTPasswd identity provider.

    Args:
        idp_name: The name of the identity provider to add.
        secret_name: The name of the secret containing the htpasswd file.
    """
    _, stdout, _ = run_command(command=["oc", "get", "oauth", "cluster", "-o", "json"], check=True)
    oauth_config = json.loads(stdout)
    oauth_config["spec"]["identityProviders"].append({
        "name": idp_name,
        "mappingMethod": "claim",
        "type": "HTPasswd",
        "htpasswd": {"fileData": {"name": secret_name}},
    })
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
        json.dump(oauth_config, temp_file)
        temp_file.flush()
        run_command(command=["oc", "replace", "-f", temp_file.name], check=True)
        os.unlink(temp_file.name)


def _remove_idp_from_oauth(idp_name: str) -> None:
    """
    Removes a specified identity provider from the OpenShift OAuth configuration.

    Args:
        idp_name: The name of the identity provider to remove.
    """
    _, stdout, _ = run_command(command=["oc", "get", "oauth", "cluster", "-o", "json"], check=True)
    oauth_config = json.loads(stdout)
    oauth_config["spec"]["identityProviders"] = [
        idp for idp in oauth_config["spec"]["identityProviders"] if idp["name"] != idp_name
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
        json.dump(oauth_config, temp_file)
        temp_file.flush()
        run_command(command=["oc", "replace", "-f", temp_file.name], check=True)
        os.unlink(temp_file.name)


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
