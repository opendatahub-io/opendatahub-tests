from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.serving_runtime import ServingRuntime

from tests.model_serving.model_server.kserve.inference_service_configuration.constants import (
    ISVC_ENV_VARS,
    ORIGINAL_PULL_SECRET,
    UPDATED_PULL_SECRET,
)
from tests.model_serving.model_server.kserve.inference_service_configuration.utils import (
    update_inference_service,
)
from utilities.constants import KServeDeploymentType
from utilities.inference_utils import create_isvc
from utilities.infra import get_pods_by_isvc_label


@pytest.fixture(scope="class")
def removed_isvc_env_vars(
    request: pytest.FixtureRequest,
    unprivileged_client: DynamicClient,
    ovms_kserve_inference_service: InferenceService,
) -> Generator[InferenceService, Any, Any]:
    if isvc_predictor_spec_model_env := ovms_kserve_inference_service.instance.spec.predictor.model.get("env"):
        isvc_predictor_spec_model_env = [
            env_var for env_var in isvc_predictor_spec_model_env if env_var.to_dict() not in ISVC_ENV_VARS
        ]

        with update_inference_service(
            client=unprivileged_client,
            isvc=ovms_kserve_inference_service,
            isvc_updated_dict={"spec": {"predictor": {"model": {"env": isvc_predictor_spec_model_env}}}},
        ):
            yield ovms_kserve_inference_service

    else:
        raise ValueError(
            f"Inference service {ovms_kserve_inference_service.name} does not have env vars in predictor spec model."
        )


@pytest.fixture
def isvc_pods(
    unprivileged_client: DynamicClient, ovms_kserve_inference_service: InferenceService
) -> Generator[list[Pod], Any, Any]:
    yield get_pods_by_isvc_label(client=unprivileged_client, isvc=ovms_kserve_inference_service)


@pytest.fixture(scope="class")
def patched_isvc_replicas(
    request: pytest.FixtureRequest,
    unprivileged_client: DynamicClient,
    ovms_kserve_inference_service: InferenceService,
) -> Generator[InferenceService, Any, Any]:
    with update_inference_service(
        client=unprivileged_client,
        isvc=ovms_kserve_inference_service,
        isvc_updated_dict={
            "spec": {
                "predictor": {
                    "maxReplicas": request.param["max-replicas"],
                    "minReplicas": request.param["min-replicas"],
                }
            }
        },
        wait_for_new_pods=request.param["wait-for-new-pods"],
    ):
        yield ovms_kserve_inference_service


@pytest.fixture(scope="class")
def model_car_raw_inference_service_with_pull_secret(
    request: pytest.FixtureRequest,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    serving_runtime_from_template: ServingRuntime,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=unprivileged_client,
        name="model-car-raw",
        namespace=unprivileged_model_namespace.name,
        runtime=serving_runtime_from_template.name,
        storage_uri=request.param["storage-uri"],
        model_format=serving_runtime_from_template.instance.spec.supportedModelFormats[0].name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        image_pull_secrets=[ORIGINAL_PULL_SECRET],
        wait_for_predictor_pods=False,  # Until modelcar initContainer completed, other containers may have Error status
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def updated_isvc_pull_secret(
    request: pytest.FixtureRequest,
    unprivileged_client: DynamicClient,
    model_car_raw_inference_service_with_pull_secret: InferenceService,
) -> Generator[InferenceService, Any, Any]:
    with update_inference_service(
        client=unprivileged_client,
        isvc=model_car_raw_inference_service_with_pull_secret,
        isvc_updated_dict={"spec": {"predictor": {"imagePullSecrets": [{"name": UPDATED_PULL_SECRET}]}}},
    ):
        yield model_car_raw_inference_service_with_pull_secret


@pytest.fixture(scope="class")
def updated_isvc_remove_pull_secret(
    request: pytest.FixtureRequest,
    unprivileged_client: DynamicClient,
    model_car_raw_inference_service_with_pull_secret: InferenceService,
) -> Generator[InferenceService, Any, Any]:
    with update_inference_service(
        client=unprivileged_client,
        isvc=model_car_raw_inference_service_with_pull_secret,
        isvc_updated_dict={
            "spec": {
                "predictor": {
                    "imagePullSecrets": None  # Explicitly remove the field
                }
            }
        },
    ):
        yield model_car_raw_inference_service_with_pull_secret
