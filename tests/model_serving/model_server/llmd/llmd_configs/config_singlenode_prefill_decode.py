"""Single-node prefill/decode disaggregation with scheduler and NixlConnector KV transfer.

Controller topology model:
- Prefill != nil → P/D mode (separate prefill + decode Deployments)
- Worker != nil  → multi-node mode (LeaderWorkerSet instead of Deployments)
- These are orthogonal: single-node P/D, multi-node P/D, or non-P/D are all valid

This config covers single-node P/D: Prefill is set, Worker is nil → standard Deployments.
Physical co-location on the same host is via pod affinity (user responsibility), not controller.

No custom scheduler config is specified. The controller auto-generates the full P/D
EndpointPickerConfig with all disaggregation plugins when spec.prefill != nil.

Fast image variants live in ``config_fast_image.py`` alongside other fast configs.
"""

from utilities.constants import Labels

from .config_models import TinyLlamaOciGpuConfig


class SingleNodePrefillDecodeConfig(TinyLlamaOciGpuConfig):
    """Single-node GPU with disaggregated Prefill/Decode and NixlConnector KV transfer.

    Requires 2 NVIDIA GPUs on a single node (1 decode + 1 prefill).
    Uses NixlConnector for KV cache transfer over UCX (NVLink/PCIe on same node).
    """

    name = "llmisvc-singlenode-pd"
    replicas = 1

    # GPU requirements
    min_nodes = 1
    min_gpus_per_node = 2
    supported_accelerators = (Labels.Nvidia.NVIDIA_COM_GPU,)
    supported_topology = "workload-single-node-pd"

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
    def router_config(cls):
        return {
            "scheduler": {},
            "route": {},
            "gateway": {},
        }

    @classmethod
    def prefill_config(cls):
        return {
            "replicas": 1,
            "template": {
                "affinity": {
                    "podAffinity": {
                        "requiredDuringSchedulingIgnoredDuringExecution": [
                            {
                                "labelSelector": {
                                    "matchExpressions": [
                                        {"key": "llm-d.ai/role", "operator": "In", "values": ["decode"]},
                                        {"key": "app.kubernetes.io/name", "operator": "In", "values": [cls.name]},
                                    ]
                                },
                                "topologyKey": "kubernetes.io/hostname",
                            }
                        ]
                    }
                },
                "containers": [
                    {
                        "name": "main",
                        "env": cls.container_env(),
                        "resources": cls.container_resources(),
                        "livenessProbe": cls.liveness_probe(),
                    }
                ],
            },
        }
