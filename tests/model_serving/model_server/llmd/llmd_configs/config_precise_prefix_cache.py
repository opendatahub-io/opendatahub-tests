"""Precise prefix cache configuration for single-node LLMInferenceService."""

import json

import yaml

from .config_models import TinyLlamaHfGpuConfig


class PrecisePrefixCacheConfig(TinyLlamaHfGpuConfig):
    """Single-node precise prefix cache — TinyLlama via HuggingFace, 2 GPU replicas."""

    name = "llmisvc-precise-prefix"
    # The precise-prefix-cache-scorer scheduler plugin downloads the tokenizer from
    # HuggingFace using the model name. It must be the full HF repo ID, not an alias.
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    replicas = 2
    block_size = 64
    hash_algo = "sha256_cbor"
    hash_seed = "42"
    enable_auth = True

    @classmethod
    def container_env(cls):
        kv_events_config = {
            "enable_kv_cache_events": True,
            "publisher": "zmq",
            "endpoint": f"tcp://{cls.name}-epp-service:5557",
            "topic": "kv@$(POD_IP):8000@$(MODEL_NAME)",
        }
        return [
            {
                "name": "POD_IP",
                "valueFrom": {"fieldRef": {"apiVersion": "v1", "fieldPath": "status.podIP"}},
            },
            {"name": "MODEL_NAME", "value": cls.model_name},
            {"name": "VLLM_LOGGING_LEVEL", "value": "DEBUG"},
            {"name": "CUDA_LAUNCH_BLOCKING", "value": "1"},
            {"name": "PYTHONHASHSEED", "value": cls.hash_seed},
            {
                "name": "VLLM_ADDITIONAL_ARGS",
                "value": (
                    f"--enable-prefix-caching "
                    f"--prefix-caching-hash-algo {cls.hash_algo} "
                    f"--block-size {cls.block_size} "
                    f"--kv-events-config '{json.dumps(kv_events_config)}'"
                ),
            },
        ]

    @classmethod
    def _scheduler_config(cls):
        """EndpointPickerConfig — precise prefix cache with KV block index tracking."""
        return {
            "apiVersion": "inference.networking.x-k8s.io/v1alpha1",
            "kind": "EndpointPickerConfig",
            "plugins": [
                {"type": "single-profile-handler"},
                {
                    "type": "precise-prefix-cache-scorer",
                    "parameters": {
                        "kvEventsConfig": {"zmqEndpoint": "tcp://*:5557", "topicFilter": "kv"},
                        "indexerConfig": {
                            "tokenProcessorConfig": {
                                "blockSize": cls.block_size,
                                "hashSeed": cls.hash_seed,
                            },
                            "kvBlockIndexConfig": {
                                "enableMetrics": True,
                                "metricsLoggingInterval": 60000000000,
                            },
                            "tokenizersPoolConfig": {
                                "hf": {"tokenizersCacheDir": "/mnt/tokenizers"},
                            },
                        },
                    },
                },
                {"type": "load-aware-scorer"},
                {"type": "max-score-picker"},
            ],
            "schedulingProfiles": [
                {
                    "name": "default",
                    "plugins": [
                        {"pluginRef": "precise-prefix-cache-scorer", "weight": 2.0},
                        {"pluginRef": "load-aware-scorer", "weight": 1.0},
                        {"pluginRef": "max-score-picker"},
                    ],
                }
            ],
        }

    @classmethod
    def _scheduler_container(cls):
        """Scheduler container with ZMQ ports and tokenizer volume mounts."""
        return {
            "name": "main",
            "ports": [
                {"name": "grpc", "containerPort": 9002, "protocol": "TCP"},
                {"name": "grpc-health", "containerPort": 9003, "protocol": "TCP"},
                {"name": "metrics", "containerPort": 9090, "protocol": "TCP"},
                {"name": "zmq", "containerPort": 5557, "protocol": "TCP"},
            ],
            "env": [{"name": "HF_HOME", "value": "/mnt/tokenizers"}],
            "volumeMounts": [{"name": "tokenizers", "mountPath": "/mnt/tokenizers", "readOnly": False}],
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
                "--model-server-metrics-https-insecure-skip-verify",
                "--cert-path",
                "/var/run/kserve/tls",
                "--config-text",
                yaml.dump(cls._scheduler_config()),
            ],
        }

    @classmethod
    def router_config(cls):
        return {
            "scheduler": {
                "template": {
                    "volumes": [{"name": "tokenizers", "emptyDir": {}}],
                    "containers": [cls._scheduler_container()],
                }
            },
            "route": {},
            "gateway": {},
        }
