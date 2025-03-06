from typing import Any, Generator

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime

from utilities.constants import KServeDeploymentType, ModelName, ModelStoragePath
from utilities.inference_utils import create_isvc


@pytest.fixture(scope="class")
def s3_flan_small_hf_caikit_serverless_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    serving_runtime_from_template: ServingRuntime,
    models_endpoint_s3_secret: Secret,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name=f"{ModelName.FLAN_T5_SMALL}",
        namespace=model_namespace.name,
        runtime=serving_runtime_from_template.name,
        storage_key=models_endpoint_s3_secret.name,
        storage_path=ModelStoragePath.FLAN_T5_SMALL_HF,
        model_format=serving_runtime_from_template.instance.spec.supportedModelFormats[0].name,
        deployment_mode=KServeDeploymentType.SERVERLESS,
        external_route=True,
    ) as isvc:
        yield isvc
