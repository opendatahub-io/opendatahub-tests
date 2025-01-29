from typing import Any, Generator, List

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.node import Node
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from ocp_resources.serving_runtime import ServingRuntime

from tests.model_serving.model_server.utils import create_isvc
from utilities.constants import KServeDeploymentType
from utilities.general import download_model_data
from utilities.infra import (
    get_pods_by_isvc_label,
    wait_for_inference_deployment_replicas,
)


@pytest.fixture(scope="session")
def nodes(admin_client: DynamicClient) -> list[Node]:
    return list(Node.get(dyn_client=admin_client))


@pytest.fixture(scope="session")
def nvidia_gpu_nodes(nodes: list[Node]) -> list[Node]:
    return [node for node in nodes if "nvidia.com/gpu.present" in node.labels.keys()]


@pytest.fixture(scope="session")
def skip_if_no_gpu_nodes(nvidia_gpu_nodes):
    if len(nvidia_gpu_nodes) < 2:
        pytest.skip("Multi-node tests can only run on a Cluster with at least 2 GPU Worker nodes")


@pytest.fixture(scope="class")
def models_bucket_downloaded_model_data(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    models_s3_bucket_name: str,
    model_pvc: PersistentVolumeClaim,
    aws_secret_access_key: str,
    aws_access_key_id: str,
    models_s3_bucket_endpoint: str,
    models_s3_bucket_region: str,
) -> str:
    return download_model_data(
        admin_client=admin_client,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        model_namespace=model_namespace.name,
        model_pvc_name=model_pvc.name,
        bucket_name=models_s3_bucket_name,
        aws_endpoint_url=models_s3_bucket_endpoint,
        aws_default_region=models_s3_bucket_region,
        model_path=request.param["model-dir"],
    )


@pytest.fixture(scope="class")
def multi_node_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    serving_runtime_from_template: ServingRuntime,
    model_pvc: PersistentVolumeClaim,
    models_bucket_downloaded_model_data: str,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name=request.param["name"],
        namespace=model_namespace.name,
        runtime=serving_runtime_from_template.name,
        storage_uri=f"pvc://{model_pvc.name}/{models_bucket_downloaded_model_data}",
        model_format=serving_runtime_from_template.instance.spec.supportedModelFormats[0].name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        autoscaler_mode="external",
        multi_node_worker_spec={},
        wait_for_predictor_pods=False,
    ) as isvc:
        wait_for_inference_deployment_replicas(
            client=admin_client,
            isvc=isvc,
            deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
            expected_num_deployments=2,
        )
        yield isvc


@pytest.fixture(scope="class")
def multi_node_predictor_pods_scope_class(
    admin_client: DynamicClient,
    multi_node_inference_service: InferenceService,
) -> List[Pod]:
    return get_pods_by_isvc_label(
        client=admin_client,
        isvc=multi_node_inference_service,
    )
