from typing import Any, Generator

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime
from pytest_testconfig import config as py_config

from tests.model_serving.model_server.components.kserve_dsc_deployment_mode.utils import (
    patch_deployment_in_dsc,
)
from utilities import constants
from utilities.constants import ModelAndFormat
from utilities.inference_utils import create_isvc


@pytest.fixture(scope="class")
def inferenceservice_config_cm(admin_client: DynamicClient) -> ConfigMap:
    return ConfigMap(
        client=admin_client,
        name="inferenceservice-config",
        namespace=py_config["applications_namespace"],
    )


@pytest.fixture(scope="class")
def patched_deployment_in_dsc(
    request: FixtureRequest,
    dsc_resource: DataScienceCluster,
    inferenceservice_config_cm: ConfigMap,
) -> Generator[DataScienceCluster, Any, Any]:
    if development_mode := request.param.get("updated-deployment-mode"):
        spec_key = "defaultDeploymentMode"
        nested_path = ["defaultDeploymentMode"]
        config_key = "deploy"
        expected_value = development_mode
    elif service_config := request.param.get("raw-deployment-service-config"):
        spec_key = "rawDeploymentServiceConfig"
        nested_path = ["serviceClusterIPNone"]
        config_key = "service"
        expected_value = service_config == constants.DscComponents.RawDeploymentServiceConfig.HEADLESS

    else:
        raise ValueError(f"Unsupported value: {request.param}")

    patch_generator = patch_deployment_in_dsc(
        dsc_resource=dsc_resource,
        config_map=inferenceservice_config_cm,
        spec_key=spec_key,
        config_key=config_key,
        expected_value=expected_value,
        nested_key_path=nested_path,
    )
    yield from patch_generator


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


@pytest.fixture(scope="class")
def restarted_inference_pod(ovms_inference_service: InferenceService) -> Pod:
    def _get_pod(isvc: InferenceService) -> Pod:
        """
        Get the pod of the inference service.

        Args:
            isvc (InferenceService): Inference service object.

        Returns:
            Pod: The pod of the inference service.

        """
        pod_kwargs = {
            "dyn_client": ovms_inference_service.client,
            "namespace": ovms_inference_service.namespace,
            "label_selector": f"{ovms_inference_service.ApiGroup.SERVING_KSERVE_IO}/"
            f"inferenceservice={ovms_inference_service.name}",
        }

        if orig_pods := list(Pod.get(**pod_kwargs)):
            if len(orig_pods) != 1:
                raise ValueError(f"Expected 1 pod, got {len(orig_pods)}")

            return orig_pods[0]

        else:
            raise ValueError("Expected at least 1 pod")

    orig_pod = _get_pod(isvc=ovms_inference_service)
    orig_pod.delete(wait=True)

    return _get_pod(isvc=ovms_inference_service)
