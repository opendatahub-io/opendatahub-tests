import shlex
import base64
from typing import Optional, Generator
from urllib.parse import urlparse

from ocp_resources.pod import Pod
from kubernetes.dynamic.client import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.inference_service import InferenceService
from pyhelper_utils.shell import run_command
from simple_logger.logger import get_logger


LOGGER = get_logger(name=__name__)


class ProtocolNotSupported(Exception):
    def __init__(self, protocol: str):
        self.protocol = protocol
    def __str__(self) -> str:
        return f"Protocol {self.protocol} is not supported"
    
class MissingStorageArgument(Exception):
    def __init__(
            self, 
            storageUri: Optional[str], 
            storage_key: Optional[str], 
            storage_path: Optional[str],
        ):
        self.storageUri=storageUri
        self.storage_key=storage_key
        self.storage_path=storage_path

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


def get_flan_pod(client: DynamicClient, namespace: str, name_prefix: str) -> Pod:
    for pod in Pod.get(dyn_client=client, namespace=namespace):
        if name_prefix + "-predictor" in pod.name:
            return pod

    raise ResourceNotFoundError(f"No flan predictor pod found in namespace {namespace}")


def curl_from_pod(
    isvc: InferenceService,
    pod: Pod,
    endpoint: str,
    protocol: str = "http",
) -> str:
    if protocol == "http":
        parsed = urlparse(isvc.instance.status.address.url)
        host = parsed._replace(scheme="http").geturl()

    elif protocol == "https":
        host = isvc.instance.status.address.url
    else:
        raise ProtocolNotSupported(protocol)

    return pod.execute(command=shlex.split(f"curl -k {host}/{endpoint}"), ignore_rc=True)


def create_sidecar_pod(
    admin_client: DynamicClient,
    namespace: str,
    istio: bool,
    pod_name: str,
) -> Generator[Pod,None,None]:
    cmd = f"oc run {pod_name} -n {namespace} --image=registry.access.redhat.com/rhel7/rhel-tools"
    if istio:
        cmd = f'{cmd} --annotations=sidecar.istio.io/inject="true"'

    cmd += " -- sleep infinity"

    _, _, err = run_command(command=shlex.split(cmd), check=False)
    if err:
        LOGGER.info(msg=err)

    with Pod(
        name=pod_name,
        namespace=namespace,
        client=admin_client,
    ) as p:
        p.wait_for_status(status="Running")
        p.wait_for_condition(condition="Ready", status="True")
        yield p


def b64_encoded_string(string_to_encode: str) -> str:
    # encodes the string to bytes-like, encodes the bytes-like to base 64, decodes the b64 to a string and returns it
    # needed for openshift resources expecting b64 encoded values in the yaml
    return base64.b64encode(string_to_encode.encode()).decode()
