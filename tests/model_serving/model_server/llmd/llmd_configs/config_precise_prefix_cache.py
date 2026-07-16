"""Precise prefix cache configuration for single-node LLMInferenceService."""

import json
import textwrap

import yaml

from .config_models import TinyLlamaHfGpuConfig, TinyLlamaOciGpuConfig


class PrecisePrefixCacheScorerConfig(TinyLlamaHfGpuConfig):
    """precise-prefix-cache-scorer plugin (inference.networking.x-k8s.io/v1alpha1).

    TinyLlama via HuggingFace, 2 GPU replicas. The scorer plugin handles tokenization,
    KV block indexing, and scoring in a single plugin. EPP binds a global ZMQ socket
    and vLLM pods connect to it.
    """

    name = "llmisvc-precise-prefix-scorer"
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    replicas = 2
    min_gpus_per_node = 2
    block_size = 64
    hash_algo = "sha256_cbor"
    hash_seed = "42"
    enable_auth = True
    wait_timeout = 720

    @classmethod
    def container_env(cls) -> list[dict]:
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
    def _scheduler_config(cls) -> dict:
        """EndpointPickerConfig dict — legacy precise-prefix-cache-scorer plugin.

        Returns a dict consumed by yaml.dump() in _scheduler_container().
        """
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
    def _scheduler_container(cls) -> dict:
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
    def router_config(cls) -> dict:
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


class PrecisePrefixCacheProducerConfig(TinyLlamaOciGpuConfig):
    """precise-prefix-cache-producer plugin (llm-d.ai/v1alpha1).

    TinyLlama via OCI, 2 GPU replicas. Tokenization handled by token-producer,
    KV indexing by precise-prefix-cache-producer, scoring by prefix-cache-scorer.
    EPP discovers pod IPs from InferencePool endpoints and dials each vLLM pod's
    ZMQ socket directly.
    """

    name = "llmisvc-precise-prefix-producer"
    replicas = 2
    min_gpus_per_node = 2
    block_size = 64
    hash_algo = "sha256_cbor"
    hash_seed = "42"
    enable_auth = True
    wait_timeout = 720

    @classmethod
    def container_env(cls) -> list[dict]:
        kv_events_config = {
            "enable_kv_cache_events": True,
            "publisher": "zmq",
            "endpoint": "tcp://*:5557",
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
    def _scheduler_config(cls) -> str:
        """EndpointPickerConfig YAML string — precise-prefix-cache-producer + prefix-cache-scorer.

        Returns a raw YAML string to preserve Go template variables.
        """
        return textwrap.dedent(f"""\
            apiVersion: llm-d.ai/v1alpha1
            kind: EndpointPickerConfig
            plugins:
              - type: single-profile-handler
              - type: token-producer
                parameters:
                  modelName: {cls.model_name}
                  vllm:
                    url: https://{cls.name}-kserve-workload-svc.{{{{ .ObjectMeta.Namespace }}}}.svc:8000
              - type: endpoint-notification-source
              - type: metrics-data-source
              - type: core-metrics-extractor
              - type: precise-prefix-cache-producer
                parameters:
                  tokenProcessorConfig:
                    blockSize: {cls.block_size}
                    hashSeed: "{cls.hash_seed}"
                  indexerConfig:
                    kvBlockIndexConfig:
                      enableMetrics: true
                      metricsLoggingInterval: 60000000000
                  kvEventsConfig:
                    topicFilter: kv
              - type: prefix-cache-scorer
                parameters:
                  prefixMatchInfoProducerName: precise-prefix-cache-producer
              - type: load-aware-scorer
              - type: max-score-picker
            dataLayer:
              sources:
                - pluginRef: metrics-data-source
                  extractors:
                    - pluginRef: core-metrics-extractor
                - pluginRef: endpoint-notification-source
                  extractors:
                    - pluginRef: precise-prefix-cache-producer
            schedulingProfiles:
              - name: default
                plugins:
                  - pluginRef: prefix-cache-scorer
                    weight: 2.0
                  - pluginRef: load-aware-scorer
                    weight: 1.0
                  - pluginRef: max-score-picker""")

    @classmethod
    def _scheduler_container(cls) -> dict:
        """Scheduler container — no tokenizer volume needed with token-producer."""
        return {
            "name": "main",
            "ports": [
                {"name": "grpc", "containerPort": 9002, "protocol": "TCP"},
                {"name": "grpc-health", "containerPort": 9003, "protocol": "TCP"},
                {"name": "metrics", "containerPort": 9090, "protocol": "TCP"},
                {"name": "zmq", "containerPort": 5557, "protocol": "TCP"},
            ],
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
                cls._scheduler_config(),
            ],
        }

    @classmethod
    def router_config(cls) -> dict:
        return {
            "scheduler": {
                "template": {
                    "containers": [cls._scheduler_container()],
                }
            },
            "route": {},
            "gateway": {},
        }
