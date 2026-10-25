"""
Fixtures for model server image validation tests.

Creates the Model Server and Metrics team (Serving Orchestration) component
infrastructure so that deployments/pods are created and their
spec.containers[*].image values can be validated against the cluster CSV
relatedImages (registry.redhat.io, sha256 digest):
- kserve control plane deployments (odh-model-controller, kserve-controller-manager),
  deployed by the `kserve` DSC component.
- InferenceService model server pods (predictor and storage-initializer images),
  deployed by the kserve controllers.
- InferenceGraph raw deployment router pods (router image), deployed by the
  inference graph controller.
"""

from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.inference_graph import InferenceGraph
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.serving_runtime import ServingRuntime
from pytest import FixtureRequest
from pytest_testconfig import config as py_config
from timeout_sampler import TimeoutSampler

from tests.model_serving.model_server.image_validation.constant import (
    MODEL_SERVER_IMAGE_VALIDATION_NAMESPACE,
    MODEL_SERVER_RUNTIME_TEMPLATE,
    PLACEHOLDER_STORAGE_URI,
)
from utilities.constants import Annotations, KServeDeploymentType, Labels
from utilities.inference_utils import create_isvc
from utilities.infra import create_ns, get_pods_by_ig_label, wait_for_isvc_pods
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="class")
def model_server_image_validation_namespace(
    admin_client: DynamicClient,
) -> Generator[Namespace, Any, Any]:
    """A dedicated namespace for model server image validation, torn down after the tests."""
    with create_ns(
        admin_client=admin_client,
        name=MODEL_SERVER_IMAGE_VALIDATION_NAMESPACE,
        teardown=True,
    ) as ns:
        yield ns


@pytest.fixture(scope="class")
def model_server_serving_runtime(
    admin_client: DynamicClient,
    model_server_image_validation_namespace: Namespace,
) -> Generator[ServingRuntime, Any, Any]:
    """A ServingRuntime created from the OVMS kserve template."""
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name="model-server-images-runtime",
        namespace=model_server_image_validation_namespace.name,
        template_name=MODEL_SERVER_RUNTIME_TEMPLATE,
        deployment_type=KServeDeploymentType.STANDARD,
    ) as serving_runtime:
        yield serving_runtime


@pytest.fixture(scope="class")
def model_server_inference_service(
    admin_client: DynamicClient,
    model_server_image_validation_namespace: Namespace,
    model_server_serving_runtime: ServingRuntime,
) -> Generator[InferenceService, Any, Any]:
    """An InferenceService with placeholder storage so its model server Deployment/Pod is created."""
    model_format = model_server_serving_runtime.instance.spec.supportedModelFormats[0].name
    with create_isvc(
        client=admin_client,
        name="model-server-images-isvc",
        namespace=model_server_image_validation_namespace.name,
        model_format=model_format,
        runtime=model_server_serving_runtime.name,
        storage_uri=PLACEHOLDER_STORAGE_URI,
        deployment_mode=KServeDeploymentType.STANDARD,
        wait=False,
        wait_for_predictor_pods=False,
        timeout=120,
        teardown=True,
    ) as isvc:
        yield isvc


@pytest.fixture
def kserve_control_plane_pods(request: FixtureRequest, admin_client: DynamicClient) -> list[Pod]:
    """Pods of a kserve control plane deployment in the applications namespace."""
    for pods in TimeoutSampler(
        wait_timeout=300,
        sleep=5,
        func=lambda: list(
            Pod.get(
                client=admin_client,
                namespace=py_config["applications_namespace"],
                label_selector=request.param["label_selector"],
            )
        ),
    ):
        if pods:
            return pods


@pytest.fixture
def isvc_model_server_pods(
    admin_client: DynamicClient,
    model_server_inference_service: InferenceService,
    model_server_serving_runtime: ServingRuntime,
) -> Generator[list[Pod], Any, Any]:
    """Pods of the InferenceService model server (predictor and storage-initializer images)."""
    for pods in TimeoutSampler(
        wait_timeout=300,
        sleep=5,
        func=wait_for_isvc_pods,
        client=admin_client,
        isvc=model_server_inference_service,
        runtime_name=model_server_serving_runtime.name,
    ):
        if pods:
            break
    yield pods


@pytest.fixture
def inference_graph_router_pods(
    admin_client: DynamicClient,
    model_server_inference_service: InferenceService,
) -> Generator[list[Pod], Any, Any]:
    """Pods of the raw InferenceGraph router deployment (router image)."""
    with InferenceGraph(
        client=admin_client,
        name=f"{model_server_inference_service.name}-ig",
        namespace=model_server_inference_service.namespace,
        nodes={
            "root": {
                "routerType": "Sequence",
                "steps": [
                    {
                        "name": "model-server",
                        "serviceName": model_server_inference_service.name,
                    }
                ],
            }
        },
        annotations={Annotations.KserveIo.DEPLOYMENT_MODE: KServeDeploymentType.RAW_DEPLOYMENT},
        label={Labels.Kserve.NETWORKING_KSERVE_IO: "cluster-local"},
    ) as inference_graph:
        # The router Deployment/Pod is created even when the referenced InferenceService is
        # not ready; pod phase does not need to be Running for image validation.
        for pods in TimeoutSampler(
            wait_timeout=300,
            sleep=5,
            exceptions_dict={ResourceNotFoundError: []},
            func=get_pods_by_ig_label,
            client=admin_client,
            ig=inference_graph,
        ):
            if pods:
                break
        yield pods
