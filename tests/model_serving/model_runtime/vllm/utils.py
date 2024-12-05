from typing import Any, Dict
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.template import Template
from ocp_resources.serving_runtime import ServingRuntime
from simple_logger.logger import get_logger
from tests.model_serving.model_runtime.vllm.constant import vLLM_CONFIG
from utilities.constants import APPLICATIONS_NAMESPACE

LOGGER = get_logger(name=__name__)


def get_runtime_manifest(client: DynamicClient, template_name: str, deployment_type: str) -> ServingRuntime:
    # Get the model template and extract the runtime dictionary
    template = get_model_template(client=client, template_name=template_name)
    runtime_dict: Dict[str, Any] = template.instance.objects[0].to_dict()

    # Determine deployment type conditions early
    is_grpc = "grpc" in deployment_type.lower()
    is_raw = "raw" in deployment_type.lower()

    # Loop through containers and apply changes
    for container in runtime_dict["spec"]["containers"]:
        # Remove '--model' from the container args, we will pass this using isvc
        container["args"] = [arg for arg in container["args"] if "--model" not in arg]

        # Update command if deployment type is grpc
        if is_grpc or is_raw:
            container["command"][-1] = vLLM_CONFIG["commands"]["GRPC"]

        if is_grpc:
            container["ports"] = vLLM_CONFIG["port_configurations"]["grpc"]
        elif is_raw:
            container["ports"] = vLLM_CONFIG["port_configurations"]["raw"]

    return runtime_dict


def get_model_template(client: DynamicClient, template_name: str) -> Template:
    template = Template(
        client=client,
        name=template_name,
        namespace=APPLICATIONS_NAMESPACE,
    )
    if template.exists:
        return template

    raise ResourceNotFoundError(f"{template_name} template not found")
