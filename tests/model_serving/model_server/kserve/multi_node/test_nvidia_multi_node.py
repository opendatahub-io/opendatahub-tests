from typing import Any

import pytest
from simple_logger.logger import get_logger

from tests.model_serving.model_server.kserve.multi_node.constants import (
    HEAD_POD_ROLE,
    WORKER_POD_ROLE,
)
from tests.model_serving.model_server.kserve.multi_node.utils import (
    get_pods_by_isvc_generation,
    is_arg_in_model_spec,
    verify_nvidia_gpu_status,
    verify_ray_status,
)
from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import Labels, Protocols, StorageClassName
from utilities.manifests.vllm import VLLM_INFERENCE_CONFIG

pytestmark = [
    pytest.mark.rawdeployment,
    pytest.mark.usefixtures("skip_if_no_gpu_nodes", "skip_if_no_nfs_storage_class"),
    pytest.mark.model_server_gpu,
    pytest.mark.multinode,
    pytest.mark.gpu,
]


LOGGER = get_logger(name=__name__)
MAX_NUM_BATCHED_TOKENS_ARG: str = "--max-num-batched-tokens=256"


@pytest.mark.parametrize(
    "unprivileged_model_namespace, models_bucket_downloaded_model_data, model_pvc, multi_node_inference_service",
    [
        pytest.param(
            {"name": "gpu-multi-node"},
            {"model-dir": "granite-8b-code-base"},
            {
                "access-modes": "ReadWriteMany",
                "storage-class-name": StorageClassName.NFS,
                "pvc-size": "40Gi",
            },
            {"name": "multi-vllm"},
        )
    ],
    indirect=True,
)
class TestMultiNode:
    def test_multi_node_ray_status(self, multi_node_predictor_pods_scope_class):
        """Test multi node ray status"""
        verify_ray_status(pods=multi_node_predictor_pods_scope_class)

    def test_multi_node_nvidia_gpu_status(self, multi_node_predictor_pods_scope_class):
        """Test multi node ray status"""
        verify_nvidia_gpu_status(pod=multi_node_predictor_pods_scope_class[0])

    def test_multi_node_default_config(self, multi_node_serving_runtime, multi_node_predictor_pods_scope_class):
        """Test multi node inference service with default config"""
        runtime_worker_spec = multi_node_serving_runtime.instance.spec.workerSpec

        if runtime_worker_spec.tensorParallelSize != 1 or runtime_worker_spec.pipelineParallelSize != 2:
            pytest.fail(f"Multinode runtime default worker spec is not as expected, {runtime_worker_spec}")

    def test_multi_node_pods_distribution(self, multi_node_predictor_pods_scope_class, nvidia_gpu_nodes):
        """Verify multi node pods are distributed between cluster GPU nodes"""
        pods_nodes = {pod.node.name for pod in multi_node_predictor_pods_scope_class}
        assert len(multi_node_predictor_pods_scope_class) == len(pods_nodes), (
            "Pods are not distributed between cluster GPU nodes"
        )

        assert pods_nodes.issubset({node.name for node in nvidia_gpu_nodes}), "Pods not running on GPU nodes"

    def test_multi_node_basic_internal_inference(self, multi_node_inference_service):
        """Test multi node basic internal inference"""
        verify_inference_response(
            inference_service=multi_node_inference_service,
            inference_config=VLLM_INFERENCE_CONFIG,
            inference_type="completions",
            protocol=Protocols.HTTP,
            use_default_query=True,
        )

    @pytest.mark.tls
    def test_tls_secret_exists_in_control_ns(self, multi_node_inference_service, ray_ca_tls_secret):
        """Test multi node ray ca tls secret exists in control (applications) namespace"""
        if not ray_ca_tls_secret.exists:
            pytest.fail(f"Secret {ray_ca_tls_secret.name} does not exist in {ray_ca_tls_secret.namespace} namespace")

    @pytest.mark.tls
    def test_tls_secret_exists_in_inference_ns(self, ray_tls_secret):
        """Test multi node ray tls secret exists in isvc namespace"""
        if not ray_tls_secret.exists:
            pytest.fail(f"Secret {ray_tls_secret.name} does not exist")

    @pytest.mark.tls
    def test_cert_files_exist_in_pods(self, multi_node_predictor_pods_scope_class):
        """Test multi node cert files exist in pods"""
        missing_certs_pods = []
        for pod in multi_node_predictor_pods_scope_class:
            certs = pod.execute(command=["ls", "/etc/ray/tls"]).split()
            if "ca.crt" not in certs or "tls.pem" not in certs:
                missing_certs_pods.append(pod.name)

        if missing_certs_pods:
            pytest.fail(f"Missing certs in pods: {missing_certs_pods}")

    @pytest.mark.parametrize(
        "deleted_multi_node_pod",
        [pytest.param({"pod-role": HEAD_POD_ROLE})],
        indirect=True,
    )
    def test_multi_node_head_pod_deletion(self, multi_node_inference_service, deleted_multi_node_pod):
        """Test multi node when head pod is deleted"""
        verify_inference_response(
            inference_service=multi_node_inference_service,
            inference_config=VLLM_INFERENCE_CONFIG,
            inference_type="completions",
            protocol=Protocols.HTTP,
            use_default_query=True,
        )

    @pytest.mark.parametrize(
        "deleted_multi_node_pod",
        [pytest.param({"pod-role": WORKER_POD_ROLE})],
        indirect=True,
    )
    def test_multi_node_worker_pod_deletion(self, multi_node_inference_service, deleted_multi_node_pod):
        """Test multi node when worker pod is deleted"""
        verify_inference_response(
            inference_service=multi_node_inference_service,
            inference_config=VLLM_INFERENCE_CONFIG,
            inference_type="completions",
            protocol=Protocols.HTTP,
            use_default_query=True,
        )

    @pytest.mark.tls
    @pytest.mark.dependency(name="test_ray_ca_tls_secret_reconciliation")
    def test_ray_ca_tls_secret_reconciliation(self, multi_node_inference_service, ray_ca_tls_secret):
        """Test multi node ray ca tls secret reconciliation"""
        ray_ca_tls_secret.clean_up()
        ray_ca_tls_secret.wait()

    @pytest.mark.tls
    @pytest.mark.dependency(name="test_ray_tls_secret_reconciliation")
    def test_ray_tls_secret_reconciliation(self, ray_tls_secret):
        """Test multi node ray ca tls secret reconciliation"""
        ray_tls_secret.clean_up()
        ray_tls_secret.wait()

    @pytest.mark.tls
    @pytest.mark.dependency(name="test_ray_tls_deleted_on_runtime_deletion")
    def test_ray_tls_deleted_on_runtime_deletion(self, ray_tls_secret, ray_ca_tls_secret, deleted_serving_runtime):
        """Test multi node ray tls secret deletion on runtime deletion"""
        ray_tls_secret.wait_deleted()
        assert ray_ca_tls_secret.exists

    @pytest.mark.tls
    @pytest.mark.dependency(depends=["test_ray_tls_deleted_on_runtime_deletion"])
    def test_ray_tls_created_on_runtime_creation(self, ray_tls_secret, ray_ca_tls_secret):
        """Test multi node ray tls secret creation on runtime creation"""
        ray_tls_secret.wait()

    def test_multi_node_basic_external_inference(self, patched_multi_node_isvc_external_route):
        """Test multi node basic external inference"""
        verify_inference_response(
            inference_service=patched_multi_node_isvc_external_route,
            inference_config=VLLM_INFERENCE_CONFIG,
            inference_type="completions",
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )

    @pytest.mark.parametrize(
        "patched_multi_node_spec",
        [
            pytest.param({
                "spec": {
                    "workerSpec": {
                        "pipelineParallelSize": 2,
                        "tensorParallelSize": 4,
                    }
                }
            })
        ],
        indirect=True,
    )
    def test_multi_node_tensor_parallel_size_propagation(self, unprivileged_client, patched_multi_node_spec):
        """Test multi node tensor parallel size (number of GPUs per pod) propagation to pod config"""
        isvc_parallel_size = str(patched_multi_node_spec.instance.spec.predictor.workerSpec.tensorParallelSize)

        failed_pods: list[dict[str, Any]] = []

        for pod in get_pods_by_isvc_generation(client=unprivileged_client, isvc=patched_multi_node_spec):
            pod_resources = pod.instance.spec.containers[0].resources
            if not (
                isvc_parallel_size
                == pod_resources.limits[Labels.Nvidia.NVIDIA_COM_GPU]
                == pod_resources.requests[Labels.Nvidia.NVIDIA_COM_GPU]
            ):
                failed_pods.append({pod.name: pod_resources})

        if failed_pods:
            pytest.fail(f"Failed pods resources : {failed_pods}, expected tesnor parallel size {isvc_parallel_size}")

    @pytest.mark.parametrize(
        "patched_multi_node_spec",
        [
            pytest.param({
                "spec": {
                    "workerSpec": {
                        "pipelineParallelSize": 2,
                        "tensorParallelSize": 1,
                    }
                }
            })
        ],
        indirect=True,
    )
    def test_multi_node_pipeline_parallel_size_propagation(self, unprivileged_client, patched_multi_node_spec):
        """Test multi node pipeline parallel size (number of pods) propagation to pod config"""
        isvc_parallel_size = patched_multi_node_spec.instance.spec.predictor.workerSpec.pipelineParallelSize
        isvc_num_pods = get_pods_by_isvc_generation(client=unprivileged_client, isvc=patched_multi_node_spec)

        if isvc_parallel_size != len(isvc_num_pods):
            pytest.fail(
                f"Expected pipeline parallel size {isvc_parallel_size} "
                f"does not match number of pods {len(isvc_num_pods)}"
            )

    @pytest.mark.parametrize(
        "patched_multi_node_spec",
        [pytest.param({"spec": {"model": {"args": [MAX_NUM_BATCHED_TOKENS_ARG]}}})],
        indirect=True,
    )
    @pytest.mark.dependency(name="test_model_args_added_to_vllm_command")
    def test_model_args_added_to_model_spec(self, unprivileged_client, patched_multi_node_spec):
        """Test model args added to vllm command"""
        if not is_arg_in_model_spec(
            client=unprivileged_client, isvc=patched_multi_node_spec, arg=MAX_NUM_BATCHED_TOKENS_ARG
        ):
            pytest.fail(f"{MAX_NUM_BATCHED_TOKENS_ARG} model args is not set in spec")

    @pytest.mark.dependency(depends=["test_model_args_added_to_vllm_command"])
    def test_model_args_removed_from_model_spec(self, unprivileged_client, multi_node_inference_service):
        """Test model args removed from vllm command"""
        if is_arg_in_model_spec(
            client=unprivileged_client, isvc=multi_node_inference_service, arg=MAX_NUM_BATCHED_TOKENS_ARG
        ):
            pytest.fail(f"{MAX_NUM_BATCHED_TOKENS_ARG} model args is not removed from spec")
