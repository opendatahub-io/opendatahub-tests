from typing import Any, Generator

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.serving_runtime import ServingRuntime

from utilities.constants import KServeDeploymentType
from utilities.inference_utils import create_isvc


@pytest.fixture(scope="class")
def model_car_inference_service(
    request: FixtureRequest,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    serving_runtime_from_template: ServingRuntime,
) -> Generator[InferenceService, Any, Any]:
    deployment_mode = request.param.get("deployment-mode", KServeDeploymentType.SERVERLESS)
    with create_isvc(
        client=unprivileged_client,
        name=f"model-car-{deployment_mode.lower()}",
        namespace=unprivileged_model_namespace.name,
        runtime=serving_runtime_from_template.name,
        storage_uri=request.param["storage-uri"],
        model_format=serving_runtime_from_template.instance.spec.supportedModelFormats[0].name,
        deployment_mode=deployment_mode,
        external_route=request.param.get("external-route", True),
        wait_for_predictor_pods=False,  # Until modelcar initContainer completed, other containers may have Error status
    ) as isvc:
        yield isvc
