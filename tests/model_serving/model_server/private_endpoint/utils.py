import shlex
import base64

from ocp_resources.pod import Pod
from kubernetes.dynamic.client import DynamicClient
from ocp_resources.inference_service import InferenceService
from pyhelper_utils.shell import run_command
from simple_logger.logger import get_logger


LOGGER = get_logger(name=__name__)


class FlanPodNotFoundError(Exception):
    pass


class ProtocolNotSupported(Exception):
    def __init__(self, protocol: str):
        self.protocol = protocol
        self.message = f"Protocol {protocol} is not supported"
        super().__init__(self.message)


def get_flan_pod(client: DynamicClient, namespace: str, name_prefix: str) -> Pod:
    for pod in Pod.get(dyn_client=client, namespace=namespace):
        if name_prefix + "-predictor" in pod.name:
            return pod

    raise FlanPodNotFoundError(f"No flan predictor pod found in namespace {namespace}")


def curl_from_pod(
    isvc: InferenceService,
    pod: Pod,
    endpoint: str,
    protocol: str = "http",
) -> str:
    if protocol == "http":
        tmp = isvc.instance.status.address.url
        host = "http://" + tmp.split("://")[1]

    elif protocol == "https":
        host = isvc.instance.status.address.url
    else:
        raise ProtocolNotSupported(protocol)

    return pod.execute(command=shlex.split(f"curl -k {host}/{endpoint}"), ignore_rc=True)


def create_sidecar_pod(admin_client, namespace, istio, pod_name):
    cmd = f"oc run {pod_name} -n {namespace} --image=registry.access.redhat.com/rhel7/rhel-tools"
    if istio:
        cmd = f'{cmd} --annotations=sidecar.istio.io/inject="true"'

    cmd += " -- sleep infinity"

    _, _, err = run_command(command=shlex.split(cmd), check=False)
    if err:
        # pytest.fail(f"Failed on {err}")
        LOGGER.info(msg=err)

    pod = Pod(name=pod_name, namespace=namespace, client=admin_client)
    pod.wait_for_status(status="Running")
    pod.wait_for_condition(condition="Ready", status="True")
    return pod


def b64_encoded_string(string_to_encode: str) -> str:
    # encodes the string to bytes-like, encodes the bytes-like to base 64, decodes the b64 to a string and returns it
    # needed for openshift resources expecting b64 encoded values in the yaml
    return base64.b64encode(string_to_encode.encode()).decode()
