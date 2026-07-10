import copy
from collections.abc import Generator
from typing import cast

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime
from pytest import FixtureRequest

from tests.model_serving.model_runtime.mlserver.constant import PREDICT_RESOURCES
from utilities.constants import KServeDeploymentType, RuntimeTemplates, Timeout
from utilities.inference_utils import create_isvc
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="class")
def negative_mlserver_serving_runtime(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    mlserver_runtime_image: str,
) -> Generator[ServingRuntime]:
    """MLServer ServingRuntime for negative test scenarios."""
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name="mlserver-neg-runtime",
        namespace=model_namespace.name,
        template_name=RuntimeTemplates.MLSERVER,
        deployment_type=request.param.get("deployment_mode", KServeDeploymentType.STANDARD),
        runtime_image=mlserver_runtime_image,
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def negative_mlserver_isvc_no_wait(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    negative_mlserver_serving_runtime: ServingRuntime,
    s3_models_storage_uri: str,
    mlserver_model_service_account: ServiceAccount,
) -> Generator[InferenceService]:
    """MLServer ISVC that does not wait for readiness — used to inspect failure conditions."""
    params = request.param
    resources = copy.deepcopy(cast(dict[str, dict[str, str]], PREDICT_RESOURCES["resources"]))

    with create_isvc(
        client=admin_client,
        name=params["name"],
        namespace=model_namespace.name,
        runtime=negative_mlserver_serving_runtime.name,
        storage_uri=s3_models_storage_uri,
        model_format=negative_mlserver_serving_runtime.instance.spec.supportedModelFormats[0].name,
        model_service_account=mlserver_model_service_account.name,
        deployment_mode=params.get("deployment_mode", KServeDeploymentType.STANDARD),
        external_route=params.get("enable_external_route", False),
        resources=resources,
        wait=False,
        wait_for_predictor_pods=False,
        timeout=Timeout.TIMEOUT_2MIN,
    ) as isvc:
        yield isvc
