"""Multinode MoE with DP+EP parallelism and Prefill/Decode disaggregation.

Combines multinode data-parallel + expert-parallel (Worker + Parallelism) with
disaggregated Prefill/Decode (Prefill != nil). The controller creates two LWS
groups — one for decode pods and one for prefill pods — each spanning multiple
nodes.

KV cache transfer between prefill and decode pods uses NixlConnector. The
controller auto-generates the full P/D EndpointPickerConfig with disaggregation
plugins when spec.prefill != nil.
"""

from utilities.constants import Labels

from .config_models import Qwen3MoeDummyGpuConfig


class MultinodeMoeDpEPPrefillDecodeConfig(Qwen3MoeDummyGpuConfig):
    """Multinode MoE with DP+EP and disaggregated Prefill/Decode.

    Deploys across 2 GPU nodes, each with 2 GPUs (1 decode + 1 prefill).
    data=2 distributes inference across nodes, expert=True enables MoE expert
    parallelism. Prefill pods handle KV cache computation and transfer via
    NixlConnector; decode pods serve end-user traffic.
    """

    name = "llmisvc-multinode-moe-dp-ep-pd"
    replicas = 1
    min_nodes = 2
    min_gpus_per_node = 2
    wait_timeout = 900
    supported_accelerators = (Labels.Nvidia.NVIDIA_COM_GPU,)
    supported_topology = "workload-multi-node-data-parallel-pd"

    # 2 decode (LWS leader+worker) + 2 prefill (LWS leader+worker)
    expected_vllm_pod_count = 4
    # Only decode pods are InferencePool members
    expected_inference_pool_pod_count = 2

    @classmethod
    def container_resources(cls):
        return {
            "limits": {"cpu": "2", "memory": "64Gi", "nvidia.com/gpu": "1"},
            "requests": {"cpu": "1", "memory": "32Gi", "nvidia.com/gpu": "1"},
        }

    @classmethod
    def container_env(cls):
        return super().container_env() + [
            {
                "name": "VLLM_ADDITIONAL_ARGS",
                "value": '--kv_transfer_config \'{"kv_connector":"NixlConnector","kv_role":"kv_both"}\'',
            },
            {
                "name": "VLLM_NIXL_SIDE_CHANNEL_HOST",
                "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}},
            },
        ]

    @classmethod
    def readiness_probe(cls):
        return {
            "httpGet": {"path": "/health", "port": 8000, "scheme": "HTTPS"},
            "initialDelaySeconds": 180,
            "periodSeconds": 10,
            "timeoutSeconds": 5,
            "failureThreshold": 12,
        }

    @classmethod
    def router_config(cls):
        return {
            "scheduler": {},
            "route": {},
            "gateway": {},
        }

    @classmethod
    def parallelism_config(cls):
        return {"data": 2, "dataLocal": 1, "expert": True}

    @classmethod
    def worker_config(cls):
        return {"containers": [{"name": "main", "resources": cls.container_resources()}]}

    @classmethod
    def prefill_config(cls):
        return {
            "replicas": 1,
            "parallelism": cls.parallelism_config(),
            "template": {
                "containers": [
                    {
                        "name": "main",
                        "env": cls.container_env(),
                        "resources": cls.container_resources(),
                        "livenessProbe": cls.liveness_probe(),
                    }
                ],
            },
            "worker": {
                "containers": [{"name": "main", "resources": cls.container_resources()}],
            },
        }
