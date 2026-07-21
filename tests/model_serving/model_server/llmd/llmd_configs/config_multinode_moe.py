"""Multinode MoE configuration for LLMInferenceService with DP+EP parallelism."""

from utilities.constants import Labels

from .config_models import Qwen3MoeDummyGpuConfig


class MultinodeMoeDpEpConfig(Qwen3MoeDummyGpuConfig):
    """Multinode MoE with data parallelism + expert parallelism (DP+EP).

    Deploys across 2 GPU nodes using LeaderWorkerSet. The controller creates a head
    pod (template) and worker pods (worker). data=2 distributes inference across
    2 nodes, expert=True enables expert parallelism for MoE routing.
    """

    name = "llmisvc-multinode-moe-dp-ep"
    replicas = 1
    min_nodes = 2
    min_gpus_per_node = 1
    wait_timeout = 900
    supported_accelerators = (Labels.Nvidia.NVIDIA_COM_GPU,)

    # 1 LWS leader (serves traffic) + 1 LWS worker (headless DP participant)
    expected_vllm_pod_count = 2
    expected_inference_pool_pod_count = 1

    # The controller auto-injects the right worker-data-parallel config when
    # it sees worker != nil + parallelism.IsDataParallel().
    # The base_refs selection just needs the accelerator-specific template
    # (default regex ^(?!.*fast-) handles that).
    supported_topology = "workload-multi-node-data-parallel"

    @classmethod
    def parallelism_config(cls):
        return {"data": 2, "dataLocal": 1, "expert": True}

    @classmethod
    def worker_config(cls):
        return {"containers": [{"name": "main", "resources": cls.container_resources()}]}
