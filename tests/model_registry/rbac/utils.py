import shlex
from typing import Any, Dict
from kubernetes.dynamic import DynamicClient
from pyhelper_utils.shell import run_command
from utilities.constants import Protocols


def get_token(user_name: str, password: str, admin_client: DynamicClient) -> str:
    """
    Get an OpenShift token for a user.
    """
    current_context = run_command(command=["oc", "config", "current-context"])[1].strip()
    command: str = (
        f"oc login  --insecure-skip-tls-verify=true {admin_client.configuration.host} -u {user_name} -p {password}"
    )
    run_command(command=shlex.split(command), hide_log_command=True)
    token = run_command(command=["oc", "whoami", "-t"])[1].strip()
    run_command(command=["oc", "config", "use-context", current_context])
    return token


def build_mr_client_args(rest_endpoint: str, token: str, author: str) -> Dict[str, Any]:
    """Builds arguments for ModelRegistryClient based on REST endpoint and token."""
    server, port = rest_endpoint.split(":")
    return {
        "server_address": f"{Protocols.HTTPS}://{server}",
        "port": port,
        "user_token": token,
        "is_secure": False,
        "author": author,
    }
