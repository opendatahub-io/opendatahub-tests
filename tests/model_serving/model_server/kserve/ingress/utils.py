import shlex
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlparse

from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from simple_logger.logger import get_logger

from utilities.constants import Protocols, Timeout
from utilities.exceptions import ProtocolNotSupportedError
from utilities.infra import get_model_route

LOGGER = get_logger(name=__name__)


def assert_ingress_status_changed(client: DynamicClient, inference_service: InferenceService) -> None:
    """
    Validates that the ingress status changes correctly after route deletion.

    Args:
        client (DynamicClient): The administrative client used to manage the model route.
        inference_service (InferenceService): The inference service whose route status is being checked.

    Raises:
        ResourceNotFoundError: If the route does not exist before or after deletion.
        AssertionError: If any of the validation checks fail.

    Returns:
        None
    """
    route = get_model_route(client=client, isvc=inference_service)
    if not route.exists:
        raise ResourceNotFoundError("Route before deletion not found: No active route is currently available.")

    initial_status = route.instance.status["ingress"][0]["conditions"][0]
    initial_host = route.host
    initial_transition_time = initial_status["lastTransitionTime"]
    initial_status_value = initial_status["status"]

    route.delete(wait=True, timeout=Timeout.TIMEOUT_1MIN)

    if not route.exists:
        raise ResourceNotFoundError("Route after deletion not found: No active route is currently available.")

    updated_status = route.instance.status["ingress"][0]["conditions"][0]
    updated_host = route.host
    updated_transition_time = updated_status["lastTransitionTime"]
    updated_status_value = updated_status["status"]

    # Collect failures instead of stopping at the first failed assertion
    failures = []

    if updated_host != initial_host:
        failures.append(f"Host mismatch: before={initial_host}, after={updated_host}")

    if updated_transition_time == initial_transition_time:
        failures.append(
            f"Transition time did not change: before={initial_transition_time}, after={updated_transition_time}"
        )

    if updated_status_value != "True":
        failures.append(f"Updated ingress status incorrect: expected=True, actual={updated_status_value}")

    if initial_status_value != "True":
        failures.append(f"Initial ingress status incorrect: expected=True, actual={initial_status_value}")

    # Assert all failures at once
    assert not failures, "Ingress status validation failed:\n" + "\n".join(failures)


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
