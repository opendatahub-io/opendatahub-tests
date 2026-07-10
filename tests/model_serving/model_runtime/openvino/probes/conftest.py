import copy
from collections.abc import Generator
from typing import Any, cast

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime
from pytest import FixtureRequest

from tests.model_serving.model_runtime.openvino.constant import PREDICT_RESOURCES
from tests.model_serving.model_runtime.openvino.probes.utils import OVMS_LIVENESS_PROBE, OVMS_READINESS_PROBE
from utilities.constants import Containers, KServeDeploymentType, RuntimeTemplates
from utilities.inference_utils import create_isvc
from utilities.infra import get_pods_by_isvc_label
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="class")
def ovms_probes_serving_runtime(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    ovms_runtime_image: str,
) -> Generator[ServingRuntime]:
    """OpenVINO ServingRuntime with readiness and liveness httpGet probes on kserve-container."""
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name="ovms-probes-runtime",
        namespace=model_namespace.name,
        template_name=RuntimeTemplates.OVMS_KSERVE,
        deployment_type=request.param.get("deployment_mode", KServeDeploymentType.STANDARD),
        runtime_image=ovms_runtime_image,
        containers={
            Containers.KSERVE_CONTAINER_NAME: {
                "readinessProbe": OVMS_READINESS_PROBE,
                "livenessProbe": OVMS_LIVENESS_PROBE,
            }
        },
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def ovms_probes_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    ovms_probes_serving_runtime: ServingRuntime,
    s3_models_storage_uri: str,
    openvino_model_service_account: ServiceAccount,
) -> Generator[InferenceService, Any, Any]:
    """OpenVINO InferenceService with probe-enabled runtime backed by S3 model storage."""
    params = request.param
    resources = copy.deepcopy(cast(dict[str, dict[str, str]], PREDICT_RESOURCES["resources"]))

    isvc_kwargs: dict[str, Any] = {
        "client": admin_client,
        "name": params["name"],
        "namespace": model_namespace.name,
        "runtime": ovms_probes_serving_runtime.name,
        "storage_uri": s3_models_storage_uri,
        "model_format": ovms_probes_serving_runtime.instance.spec.supportedModelFormats[0].name,
        "model_service_account": openvino_model_service_account.name,
        "deployment_mode": params.get("deployment_mode", KServeDeploymentType.STANDARD),
        "external_route": params.get("enable_external_route", False),
        "resources": resources,
    }

    if timeout := params.get("timeout"):
        isvc_kwargs["timeout"] = timeout

    if min_replicas := params.get("min-replicas"):
        isvc_kwargs["min_replicas"] = min_replicas

    with create_isvc(**isvc_kwargs) as isvc:
        yield isvc


@pytest.fixture
def ovms_probes_pod_resource(
    admin_client: DynamicClient,
    ovms_probes_inference_service: InferenceService,
) -> Pod:
    pods = get_pods_by_isvc_label(client=admin_client, isvc=ovms_probes_inference_service)
    if not pods:
        raise RuntimeError(f"No pods found for InferenceService {ovms_probes_inference_service.name}")
    return pods[0]
