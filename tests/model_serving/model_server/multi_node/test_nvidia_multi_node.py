import pytest

from tests.model_serving.model_server.multi_node.utils import (
    verify_nvidia_gpu_status,
    verify_ray_status,
)
from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import ModelInferenceRuntime, Protocols, StorageClassName

pytestmark = pytest.mark.usefixtures("skip_if_no_gpu_nodes", "skip_if_no_nfs_storage_class")


@pytest.mark.parametrize(
    "model_namespace, models_bucket_downloaded_model_data, model_pvc, "
    "serving_runtime_from_template, multi_node_inference_service",
    [
        pytest.param(
            {"name": "gpu-multi-node"},
            {"model-dir": "granite-8b-code-base"},
            {
                "access-modes": "ReadWriteMany",
                "storage-class-name": StorageClassName.NFS,
                "pvc-size": "40Gi",
            },
            {
                "name": "granite-runtime",
                "template-name": "vllm-multinode-runtime-template",
                "multi-model": False,
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

    def test_multi_node_default_config(self, serving_runtime_from_template, multi_node_predictor_pods_scope_class):
        """Test multi node inference service with default config"""
        runtime_worker_spec = serving_runtime_from_template.instance.spec.workerSpec

        if runtime_worker_spec.tensorParallelSize != 1 or runtime_worker_spec.pipelineParallelSize != 2:
            pytest.fail(f"Multinode runtime default worker spec is not as expected, {runtime_worker_spec}")

    def test_multi_node_pods_distribution(self, multi_node_predictor_pods_scope_class, nvidia_gpu_nodes):
        """Verify multi node pods are distributed between cluster GPU nodes"""
        pods_nodes = {pod.node.name for pod in multi_node_predictor_pods_scope_class}
        assert len(multi_node_predictor_pods_scope_class) == len(pods_nodes), (
            "Pods are not distributed between cluster GPU nodes"
        )

        assert pods_nodes.issubset({node.name for node in nvidia_gpu_nodes}), "Pods not running on GPU nodes"

    def test_multi_node_basic_inference(self, multi_node_inference_service):
        """Test multi node basic inference"""
        verify_inference_response(
            inference_service=multi_node_inference_service,
            runtime=ModelInferenceRuntime.VLLM_RUNTIME,
            inference_type="completions",
            protocol=Protocols.HTTP,
            use_default_query=True,
        )
