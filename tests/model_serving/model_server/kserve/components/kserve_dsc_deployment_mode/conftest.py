from collections.abc import Generator
from typing import Any

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime
from pytest_testconfig import config as py_config

from tests.model_serving.model_server.kserve.components.kserve_dsc_deployment_mode.utils import (
    patch_dsc_default_deployment_mode,
)
from utilities.constants import ModelAndFormat
from utilities.inference_utils import create_isvc


@pytest.fixture(scope="class")
def default_deployment_mode_in_dsc(
    request: FixtureRequest,
    dsc_resource: DataScienceCluster,
    inferenceservice_config_cm: ConfigMap,
) -> Generator[DataScienceCluster, Any, Any]:
    yield from patch_dsc_default_deployment_mode(
        dsc_resource=dsc_resource,
        inferenceservice_config_cm=inferenceservice_config_cm,
        request_default_deployment_mode=request.param["default-deployment-mode"],
    )


@pytest.fixture(scope="class")
def inferenceservice_config_cm(admin_client: DynamicClient) -> ConfigMap:
    return ConfigMap(
        client=admin_client,
        name="inferenceservice-config",
        namespace=py_config["applications_namespace"],
    )


@pytest.fixture(scope="class")
def ovms_inference_service(
    request: FixtureRequest,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    ovms_kserve_serving_runtime: ServingRuntime,
    ci_endpoint_s3_secret: Secret,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=unprivileged_client,
        name=request.param["name"],
        namespace=unprivileged_model_namespace.name,
        runtime=ovms_kserve_serving_runtime.name,
        storage_path=request.param["model-dir"],
        storage_key=ci_endpoint_s3_secret.name,
        model_format=ModelAndFormat.OPENVINO_IR,
        model_version=request.param["model-version"],
        wait_for_predictor_pods=False,
    ) as isvc:
        yield isvc
