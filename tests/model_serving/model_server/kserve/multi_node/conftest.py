from collections.abc import Generator
from typing import Any

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.node import Node
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime
from pytest_testconfig import config as py_config
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutSampler

from tests.model_serving.model_server.kserve.multi_node.constants import WORKER_POD_ROLE
from tests.model_serving.model_server.kserve.multi_node.utils import (
    delete_multi_node_pod_by_role,
    get_pods_by_isvc_generation,
)
from utilities.constants import KServeDeploymentType, Labels, ModelCarImage, Protocols, Timeout

LOGGER = get_logger(name=__name__)
from utilities.general import download_model_data
from utilities.inference_utils import create_isvc
from utilities.infra import (
    get_pods_by_isvc_label,
    verify_no_failed_pods,
    wait_for_inference_deployment_replicas,
)
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="session")
def nvidia_gpu_nodes(nodes: list[Node]) -> list[Node]:
    return [node for node in nodes if "nvidia.com/gpu.present" in node.labels]


@pytest.fixture(scope="session")
def skip_if_no_gpu_nodes(nvidia_gpu_nodes: list[Node]) -> None:
    if len(nvidia_gpu_nodes) < 2:
        pytest.skip("Multi-node tests can only run on a Cluster with at least 2 GPU Worker nodes")


@pytest.fixture(scope="class")
def models_bucket_downloaded_model_data(
    request: FixtureRequest,
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    models_s3_bucket_name: str,
    model_pvc: PersistentVolumeClaim,
    aws_secret_access_key: str,
    aws_access_key_id: str,
    models_s3_bucket_endpoint: str,
    models_s3_bucket_region: str,
) -> str:
    return download_model_data(
        client=admin_client,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        model_namespace=unprivileged_model_namespace.name,
        model_pvc_name=model_pvc.name,
        bucket_name=models_s3_bucket_name,
        aws_endpoint_url=models_s3_bucket_endpoint,
        aws_default_region=models_s3_bucket_region,
        model_path=request.param["model-dir"],
    )


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
    model_pvc: PersistentVolumeClaim,
    models_bucket_downloaded_model_data: str,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=unprivileged_client,
        name=request.param["name"],
        namespace=multi_node_serving_runtime.namespace,
        runtime=multi_node_serving_runtime.name,
        storage_uri=f"pvc://{model_pvc.name}/{models_bucket_downloaded_model_data}",
        model_format=multi_node_serving_runtime.instance.spec.supportedModelFormats[0].name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        autoscaler_mode="none",
        multi_node_worker_spec={},
        wait_for_predictor_pods=False,
    ) as isvc:
        wait_for_inference_deployment_replicas(
            client=unprivileged_client,
            isvc=isvc,
            expected_num_deployments=2,
            runtime_name=multi_node_serving_runtime.name,
        )
        yield isvc


@pytest.fixture(scope="class")
def multi_node_oci_inference_service(
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
        external_route=True,
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
    with ResourceEditor(
        patches={
            multi_node_inference_service: {
                "metadata": {"labels": {Labels.Kserve.NETWORKING_KSERVE_IO: Labels.Kserve.EXPOSED}},
            }
        }
    ):
        for sample in TimeoutSampler(
            wait_timeout=Timeout.TIMEOUT_1MIN,
            sleep=1,
            func=lambda: multi_node_inference_service.instance.status,
        ):
            if sample and sample.get("url", "").startswith(Protocols.HTTPS):
                break

        yield multi_node_inference_service


@pytest.fixture(scope="function")
def patched_multi_node_spec(
    request: FixtureRequest,
    unprivileged_client: DynamicClient,
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
        for sample in TimeoutSampler(
            wait_timeout=Timeout.TIMEOUT_10MIN,
            sleep=10,
            func=get_pods_by_isvc_generation,
            client=unprivileged_client,
            isvc=multi_node_inference_service,
        ):
            if sample:
                break
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
) -> Generator[None, Any]:
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

    wait_for_inference_deployment_replicas(
        client=unprivileged_client,
        isvc=multi_node_inference_service,
        expected_num_deployments=2,
    )

    _warmup_inference_and_wait_for_recovery(
        client=unprivileged_client,
        isvc=multi_node_inference_service,
    )


def _warmup_inference_and_wait_for_recovery(
    client: DynamicClient,
    isvc: InferenceService,
) -> None:
    """Probe the head pod with a real inference request until it succeeds.

    After a pod deletion+recovery cycle, the vLLM EngineCore may be in a corrupted
    state that only crashes on the first real inference request. The /v1/models
    endpoint returns 200 even when the engine is broken, so we must use
    /v1/completions to exercise the distributed engine path. The liveness probe
    then restarts the head pod, which takes several minutes. This function retries
    until the completions endpoint responds with 200.
    """
    probe_cmd = [
        "curl", "-s", "--max-time", "15",
        "-o", "/dev/null", "-w", "%{http_code}",
        "-H", "Content-type:application/json",
        "-d", f'{{"model":"{isvc.name}","prompt":"test","max_tokens":1}}',
        "http://localhost:8080/v1/completions",
    ]

    for sample in TimeoutSampler(
        wait_timeout=Timeout.TIMEOUT_10MIN,
        sleep=30,
        func=_probe_inference_health,
        client=client,
        isvc=isvc,
        cmd=probe_cmd,
    ):
        if sample:
            return


def _probe_inference_health(
    client: DynamicClient,
    isvc: InferenceService,
    cmd: list[str],
) -> bool:
    """Return True if the head pod's completions endpoint returns 200.

    Re-fetches pods each call to handle pod replacement during restarts.
    """
    for pod in get_pods_by_isvc_label(client=client, isvc=isvc):
        if WORKER_POD_ROLE in pod.name:
            continue

        for condition in pod.instance.status.conditions or []:
            if condition.type == "Ready" and condition.status != "True":
                LOGGER.info(f"Head pod {pod.name} not Ready yet")
                return False

        try:
            result = pod.execute(command=cmd)
            status = result.strip()
            LOGGER.info(f"Inference probe on {pod.name}: HTTP {status}")
            return status == "200"
        except Exception as exc:
            LOGGER.warning(f"Inference probe on {pod.name} failed: {exc}")
            return False

    LOGGER.warning("No head pod found")
    return False
