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
) -> str:
    """
    Curl from pod

    Args:
        isvc (InferenceService): InferenceService object
        pod (Pod): Pod object
        endpoint (str): endpoint
        protocol (str): protocol

    Returns:
        str: curl command output

    """
    if protocol not in (Protocols.HTTPS, Protocols.HTTP):
        raise ProtocolNotSupportedError(protocol)
    host = isvc.instance.status.address.url
    if protocol == "http":
        parsed = urlparse(url=host)
        host = parsed._replace(scheme="http").geturl()
    return pod.execute(command=shlex.split(f"curl -k {host}/{endpoint}"), ignore_rc=True)


@contextmanager
def create_sidecar_pod(
    client: DynamicClient,
    namespace: str,
    use_istio: bool,
    pod_name: str,
) -> Generator[Pod, Any, Any]:
    """
    Create a sidecar pod

    Args:
        client (DynamicClient): DynamicClient object
        namespace (str): namespace name
        use_istio (bool): use istio
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

    pod_kwargs = {"client": client, "name": pod_name, "namespace": namespace, "containers": containers}

    if use_istio:
        pod_kwargs.update({"annotations": {"sidecar.istio.io/inject": "true"}})

    with Pod(**pod_kwargs) as pod:
        pod.wait_for_condition(condition="Ready", status="True")
        yield pod
