from typing import Generator, Any

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_graph import InferenceGraph
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime

from utilities.constants import ModelFormat, KServeDeploymentType, ModelStoragePath
from utilities.inference_utils import create_isvc


@pytest.fixture
def dog_breed_inference_graph(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    dog_cat_inference_service: InferenceService,
    dog_breed_inference_service: InferenceService,
) -> Generator[InferenceGraph, Any, Any]:
    nodes = {
        "root": {
            "routerType": "Sequence",
            "steps": [
                {"name": "dog-cat-classifier", "serviceName": dog_cat_inference_service.name},
                {
                    "name": "dog-breed-classifier",
                    "serviceName": dog_breed_inference_service.name,
                    "data": "$request",
                    "condition": "[@this].#(outputs.0.data.1>=0)",
                },
            ],
        }
    }
    with InferenceGraph(
        client=admin_client,
        name="dog-breed-pipeline",
        namespace=model_namespace.name,
        nodes=nodes,
    ) as inference_graph:
        inference_graph.wait_for_condition(condition=inference_graph.Condition.READY, status="True")
        yield inference_graph


@pytest.fixture
def dog_cat_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    ovms_kserve_serving_runtime: ServingRuntime,
    models_endpoint_s3_secret: Secret,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name="dog-cat-classifier",
        namespace=model_namespace.name,
        runtime=ovms_kserve_serving_runtime.name,
        storage_key=models_endpoint_s3_secret.name,
        storage_path=ModelStoragePath.CAT_DOG_ONNX,
        model_format=ModelFormat.ONNX,
        deployment_mode=KServeDeploymentType.SERVERLESS,
        protocol_version="v2",
    ) as isvc:
        yield isvc


@pytest.fixture
def dog_breed_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    ovms_kserve_serving_runtime: ServingRuntime,
    models_endpoint_s3_secret: Secret,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name="dog-breed-classifier",
        namespace=model_namespace.name,
        runtime=ovms_kserve_serving_runtime.name,
        storage_key=models_endpoint_s3_secret.name,
        storage_path=ModelStoragePath.DOG_BREED_ONNX,
        model_format=ModelFormat.ONNX,
        deployment_mode=KServeDeploymentType.SERVERLESS,
        protocol_version="v2",
    ) as isvc:
        yield isvc
