import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime
# from utilities.infra import s3_endpoint_secret

from pytest import FixtureRequest
from tests.model_serving.model_runtime.vllm.utils import get_runtime_manifest

from simple_logger.logger import get_logger

LOGGER = get_logger(name=__name__)


@pytest.fixture
def ci_s3_storage_uri(request: FixtureRequest, models_s3_bucket_name: str) -> str:
    return f"s3://{models_s3_bucket_name}/{request.param['model-dir']}/"


@pytest.fixture
def model_service_account(admin_client: DynamicClient, endpoint_s3_secret: Secret) -> ServiceAccount:
    with ServiceAccount(
        client=admin_client,
        namespace=endpoint_s3_secret.namespace,
        name="models-bucket-sa",
        secrets=[{"name": endpoint_s3_secret.name}],
    ) as sa:
        yield sa


@pytest.fixture(scope="class")
def serving_runtime(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
):
    manifest = get_runtime_manifest(
        client=admin_client, template_name="vllm-runtime-template", deployment_type=request.param["deployment_type"]
    )
    manifest["metadata"]["name"] = "vllm-runtime"
    manifest["metadata"]["namespace"] = model_namespace.name
    with ServingRuntime(client=admin_client, kind_dict=manifest) as model_runtime:
        yield model_runtime
