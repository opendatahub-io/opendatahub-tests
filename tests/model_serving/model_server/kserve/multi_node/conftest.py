from typing import Any, Generator

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.node import Node
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime
from pytest_testconfig import config as py_config

from tests.model_serving.model_server.kserve.multi_node.utils import (
    delete_multi_node_pod_by_role,
    wait_for_vllm_health,
)
from timeout_sampler import TimeoutSampler

from utilities.constants import KServeDeploymentType, Labels, Protocols, Timeout, ModelCarImage
from utilities.inference_utils import create_isvc
from utilities.infra import (
    get_pods_by_isvc_label,
    verify_no_failed_pods,
    wait_for_inference_deployment_replicas,
)
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="session")
def nvidia_gpu_nodes(nodes: list[Node]) -> list[Node]:
    return [node for node in nodes if "nvidia.com/gpu.present" in node.labels.keys()]


@pytest.fixture(scope="session")
def max_gpu_per_node(nvidia_gpu_nodes: list[Node]) -> int:
    return max(
        (int(node.instance.status.allocatable.get("nvidia.com/gpu", 0)) for node in nvidia_gpu_nodes),
        default=0,
    )


@pytest.fixture(scope="session")
def skip_if_no_gpu_nodes(nvidia_gpu_nodes: list[Node]) -> None:
    if len(nvidia_gpu_nodes) < 2:
        pytest.skip("Multi-node tests can only run on a Cluster with at least 2 GPU Worker nodes")


@pytest.fixture(scope="class")
def multi_node_serving_runtime(
    request: FixtureRequest,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
) -> Generator[ServingRuntime, Any, Any]:
    with ServingRuntimeFromTemplate(
        client=unprivileged_client,
        name="vllm-multinode-runtime",  # TODO: rename servingruntime when RHOAIENG-16147 is resolved
        namespace=unprivileged_model_namespace.name,
        template_name="vllm-multinode-runtime-template",
        multi_model=False,
        enable_http=True,
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def multi_node_inference_service(
    request: FixtureRequest,
    unprivileged_client: DynamicClient,
    multi_node_serving_runtime: ServingRuntime,
) -> Generator[InferenceService, Any, Any]:
    resources = {
        "requests": {
            "cpu": "1",
            "memory": "4G",
        },
        "limits": {
            "cpu": "2",
            "memory": "12G",
        },
    }

    worker_resources = {
        "containers": [
            {
                "name": "worker-container",
                "resources": resources,
            }
        ]
    }

    with create_isvc(
        client=unprivileged_client,
        name=request.param["name"],
        namespace=multi_node_serving_runtime.namespace,
        runtime=multi_node_serving_runtime.name,
        storage_uri=ModelCarImage.GRANITE_8B_CODE_INSTRUCT,
        model_format=multi_node_serving_runtime.instance.spec.supportedModelFormats[0].name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        autoscaler_mode="none",
        resources=resources,
        multi_node_worker_spec=worker_resources,
        wait_for_predictor_pods=False,
        timeout=Timeout.TIMEOUT_30MIN,
    ) as isvc:
        wait_for_inference_deployment_replicas(
            client=unprivileged_client,
            isvc=isvc,
            expected_num_deployments=2,
            runtime_name=multi_node_serving_runtime.name,
            timeout=Timeout.TIMEOUT_15MIN,
        )
        yield isvc


@pytest.fixture(scope="class")
def multi_node_predictor_pods_scope_class(
    unprivileged_client: DynamicClient,
    multi_node_inference_service: InferenceService,
) -> list[Pod]:
    return get_pods_by_isvc_label(
        client=unprivileged_client,
        isvc=multi_node_inference_service,
    )


@pytest.fixture(scope="function")
def patched_multi_node_isvc_external_route(
    multi_node_inference_service: InferenceService,
) -> Generator[InferenceService, Any, Any]:
    multi_node_inference_service.wait_for_condition(
        condition=multi_node_inference_service.Condition.READY,
        status=multi_node_inference_service.Condition.Status.TRUE,
        timeout=Timeout.TIMEOUT_10MIN,
    )

    with ResourceEditor(
        patches={
            multi_node_inference_service: {
                "metadata": {"labels": {Labels.Kserve.NETWORKING_KSERVE_IO: Labels.Kserve.EXPOSED}},
            }
        }
    ):
        for sample in TimeoutSampler(
            wait_timeout=Timeout.TIMEOUT_5MIN,
            sleep=1,
            func=lambda: multi_node_inference_service.instance.status,
        ):
            if sample and sample.get("url", "").startswith(Protocols.HTTPS):
                break

        yield multi_node_inference_service


@pytest.fixture(scope="function")
def patched_multi_node_spec(
    request: FixtureRequest,
    multi_node_inference_service: InferenceService,
) -> Generator[InferenceService, Any, Any]:
    with ResourceEditor(
        patches={
            multi_node_inference_service: {
                "spec": {
                    "predictor": request.param["spec"],
                },
            }
        }
    ):
        yield multi_node_inference_service


@pytest.fixture()
def ray_ca_tls_secret(admin_client: DynamicClient) -> Secret:
    return Secret(
        client=admin_client,
        name="ray-ca-tls",
        namespace=py_config["applications_namespace"],
    )


@pytest.fixture()
def ray_tls_secret(unprivileged_client: DynamicClient, multi_node_inference_service: InferenceService) -> Secret:
    return Secret(
        client=unprivileged_client,
        name="ray-tls",
        namespace=multi_node_inference_service.namespace,
    )


@pytest.fixture()
def deleted_serving_runtime(
    multi_node_serving_runtime: ServingRuntime,
) -> Generator[None, Any, None]:
    multi_node_serving_runtime.clean_up()

    yield

    multi_node_serving_runtime.deploy()


@pytest.fixture()
def deleted_multi_node_pod(
    request: FixtureRequest,
    unprivileged_client: DynamicClient,
    multi_node_inference_service: InferenceService,
) -> None:
    delete_multi_node_pod_by_role(
        client=unprivileged_client,
        isvc=multi_node_inference_service,
        role=request.param["pod-role"],
    )

    verify_no_failed_pods(
        client=unprivileged_client,
        isvc=multi_node_inference_service,
        timeout=Timeout.TIMEOUT_10MIN,
    )

    wait_for_vllm_health(
        client=unprivileged_client,
        isvc=multi_node_inference_service,
    )
