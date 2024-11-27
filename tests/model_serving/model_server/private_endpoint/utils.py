import shlex
import base64
from typing import Optional, Any, Generator
from urllib.parse import urlparse
from contextlib import contextmanager

from ocp_resources.pod import Pod
from ocp_resources.deployment import Deployment
from kubernetes.dynamic.client import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError, ResourceNotUniqueError
from ocp_resources.inference_service import InferenceService
from pyhelper_utils.shell import run_command
from simple_logger.logger import get_logger


LOGGER = get_logger(name=__name__)


class ProtocolNotSupported(Exception):
    def __init__(self, protocol: str):
        self.protocol = protocol

    def __str__(self) -> str:
        return f"Protocol {self.protocol} is not supported"


class InvalidStorageArgument(Exception):
    def __init__(
        self,
        storageUri: Optional[str],
        storage_key: Optional[str],
        storage_path: Optional[str],
    ):
        self.storageUri = storageUri
        self.storage_key = storage_key
        self.storage_path = storage_path

    def __str__(self) -> str:
        msg = f"""
            You've passed the following parameters:
            "storageUri": {self.storageUri}
            "storage_key": {self.storage_key}
            "storage_path: {self.storage_path}
            In order to create a valid ISVC you need to specify either a storageUri value
            or both a storage key and a storage path.
        """
        return msg


def get_kserve_predictor_deployment(client: DynamicClient, namespace: str, name_prefix: str) -> Deployment:
    deployments = list(
        Deployment.get(
            label_selector=f"serving.kserve.io/inferenceservice={name_prefix}",
            client=client,
            namespace=namespace,
        )
    )

    if len(deployments) == 1:
        deployment = deployments[0]
        if deployment.exists:
            deployment.wait_for_replicas()
            return deployment
    elif len(deployments) > 1:
        raise ResourceNotUniqueError(f"Multiple predictor deployments found in namespace {namespace}")
    else:
        raise ResourceNotFoundError(f"Predictor deployment not found in namespace {namespace}")


def curl_from_pod(
    isvc: InferenceService,
    pod: Pod,
    endpoint: str,
    protocol: str = "http",
) -> str:
    if protocol not in ("https", "http"):
        raise ProtocolNotSupported(protocol)
    host = isvc.instance.status.address.url
    if protocol == "http":
        parsed = urlparse(host)
        host = parsed._replace(scheme="http").geturl()
    return pod.execute(command=shlex.split(f"curl -k {host}/{endpoint}"), ignore_rc=True)


@contextmanager
def create_sidecar_pod(
    admin_client: DynamicClient,
    namespace: str,
    use_istio: bool,
    pod_name: str,
) -> Generator[Pod, Any, Any]:
    cmd = f"oc run {pod_name} -n {namespace} --image=registry.access.redhat.com/rhel7/rhel-tools"
    if use_istio:
        cmd = f'{cmd} --annotations=sidecar.istio.io/inject="true"'

    cmd += " -- sleep infinity"

    _, _, err = run_command(command=shlex.split(cmd), check=False)
    if err:
        LOGGER.error(msg=err)

    pod = Pod(name=pod_name, namespace=namespace, client=admin_client)
    pod.wait_for_condition(condition="Ready", status="True")
    yield pod
    pod.clean_up()


def b64_encoded_string(string_to_encode: str) -> str:
    """Returns openshift compliant base64 encoding of a string

    encodes the input string to bytes-like, encodes the bytes-like to base 64,
    decodes the b64 to a string and returns it. This is needed for openshift
    resources expecting b64 encoded values in the yaml.

    Args:
        string_to_encode: The string to encode in base64

    Returns:
        A base64 encoded string that is compliant with openshift's yaml format
    """
    return base64.b64encode(string_to_encode.encode()).decode()
