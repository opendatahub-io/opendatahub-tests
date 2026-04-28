"""Estimated prefix cache configuration for single-node LLMInferenceService."""

import yaml

from .config_models import TinyLlamaS3GpuConfig


class EstimatedPrefixCacheConfig(TinyLlamaS3GpuConfig):
    """Single-node estimated prefix cache — TinyLlama via S3, 2 GPU replicas."""

    enable_auth = True
    name = "llmisvc-estimated-prefix"
    replicas = 2
    block_size = 64
    hash_algo = "sha256"
    hash_seed = "42"

    @classmethod
    def container_env(cls):
        return [
            {"name": "MODEL_NAME", "value": cls.model_name},
            {"name": "VLLM_LOGGING_LEVEL", "value": "DEBUG"},
            {"name": "PYTHONHASHSEED", "value": cls.hash_seed},
            {
                "name": "VLLM_ADDITIONAL_ARGS",
                "value": f"--prefix-caching-hash-algo {cls.hash_algo} --block-size {cls.block_size}",
            },
        ]

    @classmethod
    def _scheduler_config(cls):
        """EndpointPickerConfig — estimated prefix cache scorer plugin."""
        return {
            "apiVersion": "inference.networking.x-k8s.io/v1alpha1",
            "kind": "EndpointPickerConfig",
            "plugins": [
                {
                    "type": "prefix-cache-scorer",
                    "parameters": {
                        "blockSize": cls.block_size,
                        "maxPrefixBlocksToMatch": 256,
                        "lruCapacityPerServer": 31250,
                    },
                }
            ],
            "schedulingProfiles": [
                {
                    "name": "default",
                    "plugins": [{"pluginRef": "prefix-cache-scorer", "weight": 5.0}],
                }
            ],
        }

    @classmethod
    def router_config(cls):
        return {
            "scheduler": {
                "template": {
                    "containers": [
                        {
                            "name": "main",
                            "args": [
                                "--v=4",
                                "--pool-name",
                                "{{ ChildName .ObjectMeta.Name `-inference-pool` }}",
                                "--pool-namespace",
                                "{{ .ObjectMeta.Namespace }}",
                                "--pool-group",
                                "inference.networking.x-k8s.io",
                                "--zap-encoder",
                                "json",
                                "--grpc-port",
                                "9002",
                                "--grpc-health-port",
                                "9003",
                                "--secure-serving",
                                "--model-server-metrics-scheme",
                                "https",
                                "--cert-path",
                                "/var/run/kserve/tls",
                                "--config-text",
                                yaml.dump(cls._scheduler_config()),
                            ],
                        }
                    ],
                }
            },
            "route": {},
            "gateway": {},
        }
