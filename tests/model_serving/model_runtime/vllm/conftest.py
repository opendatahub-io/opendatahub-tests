import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.serving_runtime import ServingRuntime
# from utilities.infra import s3_endpoint_secret

from pytest import FixtureRequest
from tests.model_serving.model_runtime.vllm.utils import get_runtime_manifest
from tests.model_serving.model_runtime.vllm.constant import TEMPLATE_MAP

from simple_logger.logger import get_logger

LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="class")
def serving_runtime(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    supported_accelerator_type: str,
    vllm_runtime_image: str,
):
    accelerator_type = supported_accelerator_type.lower()
    template_name = TEMPLATE_MAP.get(accelerator_type, "vllm-runtime-template")
    manifest = get_runtime_manifest(
        client=admin_client,
        template_name=template_name,
        deployment_type=request.param["deployment_type"],
        runtime_image=vllm_runtime_image,
    )
    manifest["metadata"]["name"] = "vllm-runtime"
    manifest["metadata"]["namespace"] = model_namespace.name
    with ServingRuntime(client=admin_client, kind_dict=manifest) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="session")
def skip_if_no_supported_accelerator_type(supported_accelerator_type: str):
    if not supported_accelerator_type:
        pytest.skip("Accelartor type is not provide,vLLM test can not be run on CPU")
