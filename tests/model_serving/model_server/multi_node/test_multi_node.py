import pytest

from utilities.constants import StorageType

pytestmark = pytest.mark.usefixtures("skip_if_no_gpu_nodes", "skip_if_no_nfs_storage_class")


@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, model_pvc, multi_node_serving_runtime, multi_node_inference_service",
    [
        pytest.param(
            {"name": "gpu-multi-node"},
            {"model-dir": "granite-8b-code-base"},
            {"access-modes": "ReadWriteMany", "storage-class-name": StorageType.NFS, "size": "40Gi"},
            {
                "name": "granite-runtime",
                "template-name": "vllm-multinode-runtime-template ",
            },
            {"name": "multi-vllm"},
        )
    ],
    indirect=True,
)
class TestMultiNode:
    def test_multi_node_default_config(self, multi_node_inference_service):
        """Test multi node inference service with default config"""
        pass
