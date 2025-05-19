from typing import Any, Dict, Generator
from pyhelper_utils.shell import run_command
from utilities.constants import Protocols
import logging
from contextlib import contextmanager

LOGGER = logging.getLogger(__name__)


def build_mr_client_args(rest_endpoint: str, token: str, author: str) -> Dict[str, Any]:
    """
    Builds arguments for ModelRegistryClient based on REST endpoint and token.

    Args:
        rest_endpoint: The REST endpoint of the Model Registry instance.
        token: The token for the user.
        author: The author of the request.

    Returns:
        A dictionary of arguments for ModelRegistryClient.

    Note: Uses is_secure=False for testing purposes.
    """
    server, port = rest_endpoint.split(":")
    return {
        "server_address": f"{Protocols.HTTPS}://{server}",
        "port": port,
        "user_token": token,
        "is_secure": False,
        "author": author,
    }


@contextmanager
def use_context(context_name: str) -> Generator[None, None, None]:
    """
    Context manager to temporarily switch to a specific context.

    Args:
        context_name: The name of the context to switch to

    Yields:
        None

    Example:
        with use_context("my-context"):
            # Commands here will run in my-context
            run_command(["oc", "get", "pods"])
    """
    # Store current context
    _, current_context, _ = run_command(command=["oc", "config", "current-context"], check=True)
    current_context = current_context.strip()

    try:
        # Switch to the test context
        run_command(command=["oc", "config", "use-context", context_name], check=True)
        LOGGER.info(f"Switched to context: {context_name}")
        yield
    finally:
        # Restore original context
        run_command(command=["oc", "config", "use-context", current_context], check=True)
        LOGGER.info(f"Restored context: {current_context}")
