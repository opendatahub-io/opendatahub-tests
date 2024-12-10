from contextlib import contextmanager
from typing import Generator, Any, Dict
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.secret import Secret
from ocp_resources.template import Template
from ocp_resources.serving_runtime import ServingRuntime
from simple_logger.logger import get_logger
from tests.model_serving.model_runtime.vllm.constant import vLLM_CONFIG
from utilities.constants import APPLICATIONS_NAMESPACE
from tests.model_serving.model_runtime.vllm.constant import CHAT_QUERY, COMPLETION_QUERY
from utilities.plugins.constant import OpenAIEnpoints
from utilities.plugins.openai_plugin import OpenAIClient

LOGGER = get_logger(name=__name__)


def get_runtime_manifest(
    client: DynamicClient, template_name: str, deployment_type: str, runtime_image: str
) -> ServingRuntime:
    # Get the model template and extract the runtime dictionary
    template = get_model_template(client=client, template_name=template_name)
    runtime_dict: Dict[str, Any] = template.instance.objects[0].to_dict()

    # Determine deployment type conditions early
    is_grpc = "grpc" in deployment_type.lower()
    is_raw = "raw" in deployment_type.lower()

    # Loop through containers and apply changes
    for container in runtime_dict["spec"]["containers"]:
        if runtime_image:
            container["image"] = runtime_image
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


@contextmanager
def kserve_s3_endpoint_secret(
    admin_client: DynamicClient,
    name: str,
    namespace: str,
    aws_access_key: str,
    aws_secret_access_key: str,
    aws_s3_endpoint: str,
    aws_s3_region: str,
) -> Generator[Secret, None, None]:
    with Secret(
        client=admin_client,
        name=name,
        namespace=namespace,
        annotations={
            "serving.kserve.io/s3-endpoint": aws_s3_endpoint.replace("https://", ""),
            "serving.kserve.io/s3-region": aws_s3_region,
            "serving.kserve.io/s3-useanoncredential": "false",
            "serving.kserve.io/s3-verifyssl": "0",
            "serving.kserve.io/s3-usehttps": "1",
        },
        string_data={
            "AWS_ACCESS_KEY_ID": aws_access_key,
            "AWS_SECRET_ACCESS_KEY": aws_secret_access_key,
        },
        wait_for_resource=True,
    ) as secret:
        yield secret


def fetch_openai_response(  # type: ignore
    url: str,
    model_name: str,
    chat_query=CHAT_QUERY,
    completion_query=COMPLETION_QUERY,
) -> tuple[Any, list[Any], list[Any]]:
    completion_responses = []
    chat_responses = []
    inference_client = OpenAIClient(host=url, model_name=model_name, streaming=True)
    if chat_query:
        for query in chat_query:
            chat_response = inference_client.request_http(endpoint=OpenAIEnpoints.CHAT_COMPLETIONS, query=query)
            chat_responses.append(chat_response)
    if completion_query:
        for query in COMPLETION_QUERY:
            completion_response = inference_client.request_http(
                endpoint=OpenAIEnpoints.COMPLETIONS, query=query, extra_param={"max_tokens": 100}
            )
            completion_responses.append(completion_response)

    model_info = OpenAIClient.get_request_http(host=url, endpoint=OpenAIEnpoints.MODELS_INFO)
    return model_info, chat_responses, completion_responses
