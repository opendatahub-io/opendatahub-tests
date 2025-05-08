import shlex
from typing import Any, Dict
from kubernetes.dynamic import DynamicClient
from pyhelper_utils.shell import run_command
from utilities.constants import Protocols


def get_token(user_name: str, password: str, admin_client: DynamicClient) -> str:
    """
    Get an OpenShift token for a user.

    Args:
        user_name: The name of the user.
        password: The password of the user.
        admin_client: The admin client.

    Returns:
        A token for the user.

    Note: Uses --insecure-skip-tls-verify=true for testing purposes.
    """
    current_context = run_command(command=["oc", "config", "current-context"])[1].strip()
    command: str = (
        f"oc login  --insecure-skip-tls-verify=true {admin_client.configuration.host} -u {user_name} -p {password}"
    )
    run_command(command=shlex.split(command), hide_log_command=True)
    token = run_command(command=["oc", "whoami", "-t", "--insecure-skip-tls-verify=true"], hide_log_command=True)[
        1
    ].strip()
    run_command(command=["oc", "config", "use-context", current_context])
    return token


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
