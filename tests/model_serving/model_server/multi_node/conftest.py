from typing import Any, Generator

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.node import Node
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.serving_runtime import ServingRuntime

from tests.model_serving.model_server.utils import create_isvc
from utilities.constants import KServeDeploymentType
from utilities.general import download_model_data
from utilities.infra import wait_for_kserve_predictor_deployment_replicas
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="session")
def nodes(admin_client: DynamicClient) -> list[Node]:
    return list(Node.get(dyn_client=admin_client))


@pytest.fixture(scope="session")
def gpu_nodes(nodes: list[Node]) -> list[Node]:
    return [node for node in nodes if "nvidia.com/gpu.present" in node.labels.keys()]


@pytest.fixture(scope="session")
def skip_if_no_gpu_nodes(gpu_nodes):
    if len(gpu_nodes) < 2:
        pytest.skip("Multi-node tests can only run on a Cluster with at least 2 GPU Worker nodes")


@pytest.fixture(scope="class")
def downloaded_model_data(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    s3_models_storage_uri: str,
    model_pvc: PersistentVolumeClaim,
    aws_secret_access_key: str,
    aws_access_key_id: str,
) -> str:
    return download_model_data(
        admin_client=admin_client,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        storage_uri=s3_models_storage_uri,
        model_namespace=model_namespace.name,
        model_pvc_name=model_pvc.name,
    )


@pytest.fixture(scope="class")
def multi_node_serving_runtime(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[ServingRuntime, Any, Any]:
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name=request.param["name"],
        namespace=model_namespace.name,
        template_name=request.param["template-name"],
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def multi_node_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    multi_node_serving_runtime: ServingRuntime,
    model_pvc: PersistentVolumeClaim,
    downloaded_model_data: str,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name=request.param["name"],
        namespace=model_namespace.name,
        runtime=multi_node_serving_runtime.name,
        storage_uri=f"pvc://{model_pvc.name}/{downloaded_model_data}",
        model_format=multi_node_serving_runtime.instance.spec.supportedModelFormats[0].name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        autoscaler_mode="external",
        multi_node_worker_spec={},
    ) as isvc:
        wait_for_kserve_predictor_deployment_replicas(
            client=admin_client,
            isvc=isvc,
        )
        yield isvc
