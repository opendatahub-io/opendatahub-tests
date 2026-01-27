# Liveness probe for single-node configurations
LLMD_LIVENESS_PROBE = {
    "httpGet": {"path": "/health", "port": 8000, "scheme": "HTTPS"},
    "initialDelaySeconds": 240,
    "periodSeconds": 60,
    "timeoutSeconds": 60,
    "failureThreshold": 10,
}

# Common parameters for vLLM and llm-d scheduler
PREFIX_CACHE_BLOCK_SIZE = 64
PREFIX_CACHE_HASH_ALGO = "sha256_cbor"
PREFIX_CACHE_HASH_SEED = "42"

# Scheduler configuration for single-node with estimated prefix cache
ROUTER_SCHEDULER_CONFIG_ESTIMATED_PREFIX_CACHE = {
    "apiVersion": "inference.networking.x-k8s.io/v1alpha1",
    "kind": "EndpointPickerConfig",
    "plugins": [
        {
            "type": "prefix-cache-scorer",
            "parameters": {
                "blockSize": PREFIX_CACHE_BLOCK_SIZE,
                "maxPrefixBlocksToMatch": 256,
                "lruCapacityPerServer": 31250,
            },
        }
    ],
    "schedulingProfiles": [
        {
            "name": "default",
            "plugins": [
                {
                    "pluginRef": "prefix-cache-scorer",
                    "weight": 5.0,
                }
            ],
        }
    ],
}

# Scheduler configuration for single-node with precise prefix cache
ROUTER_SCHEDULER_CONFIG_PRECISE_PREFIX_CACHE = {
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
                        "blockSize": PREFIX_CACHE_BLOCK_SIZE,
                        "hashSeed": PREFIX_CACHE_HASH_SEED,
                    },
                    "kvBlockIndexConfig": {
                        "enableMetrics": True,
                        "metricsLoggingInterval": 60000000000,  # Log metrics every 60 seconds (in nanoseconds)
                    },
                    "tokenizersPoolConfig": {
                        "hf": {
                            "tokenizersCacheDir": "/mnt/tokenizers",
                        },
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
