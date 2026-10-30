import json
from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from ocp_resources.serving_runtime import ServingRuntime
from pytest_testconfig import py_config

from tests.model_serving.model_server.kserve.inference_service_lifecycle.constants import (
    ISVC_ENV_VARS,
    ISVC_LOGGER_MODE,
    ISVC_LOGGER_SINK_URL,
    ORIGINAL_PULL_SECRET,
    UPDATED_PULL_SECRET,
)
from tests.model_serving.model_server.kserve.inference_service_lifecycle.utils import (
    update_inference_service,
)
from tests.model_serving.model_server.utils import wait_for_isvc_agent_sidecar
from utilities.constants import Annotations, KServeDeploymentType, Timeout
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


@pytest.fixture(scope="function")
def patched_raw_inference_service_stop_annotation(
    request: pytest.FixtureRequest,
    ovms_raw_inference_service: InferenceService,
) -> Generator[InferenceService, Any, Any]:
    with ResourceEditor(
        patches={
            ovms_raw_inference_service: {
                "metadata": {
                    "annotations": {Annotations.KserveIo.FORCE_STOP_RUNTIME: request.param.get("stop", "false")}
                },
            }
        }
    ):
        yield ovms_raw_inference_service


@pytest.fixture(scope="class")
def logger_enabled_isvc(
    unprivileged_client: DynamicClient,
    ovms_kserve_inference_service: InferenceService,
) -> Generator[InferenceService, Any, Any]:
    """Enable inference logging on the ISVC and wait for the kserve-agent sidecar.

    Patches spec.predictor.logger the way TrustyAI enables logging on models it
    monitors; the controller rolls the predictor Deployment and the pod mutator
    webhook injects the kserve-agent sidecar into the new pod.
    """
    with update_inference_service(
        client=unprivileged_client,
        isvc=ovms_kserve_inference_service,
        isvc_updated_dict={
            "spec": {
                "predictor": {
                    "logger": {
                        "url": ISVC_LOGGER_SINK_URL,
                        "mode": ISVC_LOGGER_MODE,
                    }
                }
            }
        },
    ):
        wait_for_isvc_agent_sidecar(client=unprivileged_client, isvc=ovms_kserve_inference_service)
        ovms_kserve_inference_service.wait_for_condition(
            condition=InferenceService.Condition.READY,
            status=InferenceService.Condition.Status.TRUE,
            timeout=Timeout.TIMEOUT_10MIN,
        )
        yield ovms_kserve_inference_service


@pytest.fixture(scope="function")
def patched_kserve_agent_config(admin_client: DynamicClient) -> Generator[ConfigMap, Any, Any]:
    """Change the kserve-agent section of inferenceservice-config and restore it after.

    Simulates the agent configuration delta an operator upgrade ships (the
    inferenceservice-config ConfigMap is rendered from the upgrade payload).
    Only values read at pod-creation time by the pod mutator webhook live here,
    so the change must not affect running model pods.
    """
    kserve_cm = ConfigMap(
        client=admin_client,
        name="inferenceservice-config",
        namespace=py_config["applications_namespace"],
        ensure_exists=True,
    )
    agent_config = json.loads(kserve_cm.instance.data.agent)
    patched_cpu_request = "200m" if agent_config.get("cpuRequest") != "200m" else "300m"

    with ResourceEditor(
        patches={
            kserve_cm: {
                "metadata": {"annotations": {Annotations.OpenDataHubIo.MANAGED: "false"}},
                "data": {"agent": json.dumps({**agent_config, "cpuRequest": patched_cpu_request})},
            }
        }
    ):
        yield kserve_cm
