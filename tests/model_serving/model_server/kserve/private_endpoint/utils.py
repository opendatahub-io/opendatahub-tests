import shlex
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlparse

from kubernetes.dynamic.client import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from simple_logger.logger import get_logger

from utilities.constants import Protocols
from utilities.exceptions import ProtocolNotSupportedError

LOGGER = get_logger(name=__name__)


def curl_from_pod(
    isvc: InferenceService,
    pod: Pod,
    endpoint: str,
    protocol: str = Protocols.HTTP,
    port: int | None = None,
) -> str:
    """
    Curl from pod and return HTTP status code.

    Args:
        isvc (InferenceService): InferenceService object
        pod (Pod): Pod object
        endpoint (str): endpoint path
        protocol (str): protocol (http or https)
        port (int | None): override the port in the ISVC URL

    Returns:
        str: HTTP status code as string (e.g. "200")

    """
    if protocol not in (Protocols.HTTPS, Protocols.HTTP):
        raise ProtocolNotSupportedError(protocol)

    parsed = urlparse(url=isvc.instance.status.address.url)
    parsed = parsed._replace(scheme=protocol)
    if port:
        parsed = parsed._replace(netloc=f"{parsed.hostname}:{port}")

    url = f"{parsed.geturl()}/{endpoint}"
    cmd = shlex.split(f"curl -s -o /dev/null -w '%{{http_code}}' -k {url}")
    return pod.execute(command=cmd, ignore_rc=True)


@contextmanager
def create_curl_pod(
    client: DynamicClient,
    namespace: str,
    pod_name: str,
) -> Generator[Pod, Any, Any]:
    """
    Create a lightweight pod for running curl commands.

    Args:
        client (DynamicClient): DynamicClient object
        namespace (str): namespace name
        pod_name (str): pod name

    Returns:
        Generator[Pod, Any, Any]: pod object

    """
    containers = [
        {
            "name": pod_name,
            "image": "registry.access.redhat.com/rhel7/rhel-tools",
            "imagePullPolicy": "Always",
            "args": ["sleep", "infinity"],
            "securityContext": {
                "allowPrivilegeEscalation": False,
                "seccompProfile": {"type": "RuntimeDefault"},
                "capabilities": {"drop": ["ALL"]},
            },
        }
    ]

    with Pod(client=client, name=pod_name, namespace=namespace, containers=containers) as pod:
        pod.wait_for_condition(condition="Ready", status="True")
        yield pod
