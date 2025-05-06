import shlex
from typing import Any, ContextManager
from kubernetes.dynamic import DynamicClient
from pyhelper_utils.shell import run_command
from model_registry import ModelRegistry as ModelRegistryClient
from tests.model_registry.utils import get_endpoint_from_mr_service, get_mr_service_by_label
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


def assert_mr_client(
    user_token: str,
    admin_client: DynamicClient,
    context: ContextManager[None],
    mr_instance: Any,
    mr_namespace_name: str,
) -> None:
    """
    Assert that the Model Registry client can be created and used.
    """
    namespace_instance = admin_client.resources.get(api_version="v1", kind="Namespace").get(name=mr_namespace_name)
    svc = get_mr_service_by_label(client=admin_client, ns=namespace_instance, mr_instance=mr_instance)
    server, port = get_endpoint_from_mr_service(svc, Protocols.REST).split(":")
    with context:
        ModelRegistryClient(
            server_address=f"https://{server}",
            port=int(port),
            author="opendatahub-test",
            user_token=user_token,
            is_secure=False,
        )
