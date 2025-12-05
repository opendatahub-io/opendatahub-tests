# Liveness probe for single-node configurations
LLMD_LIVENESS_PROBE = {
    "httpGet": {"path": "/health", "port": 8000, "scheme": "HTTPS"},
    "initialDelaySeconds": 120,
    "periodSeconds": 30,
    "timeoutSeconds": 30,
    "failureThreshold": 5,
}

# Common parameters for vLLM and llm-d scheduler
PREFIX_CACHE_BLOCK_SIZE = 64
PREFIX_CACHE_HASH_ALGO = "sha256"
PREFIX_CACHE_HASH_SEED = "42"

# Scheduler configuration for single-node with precise prefix cache
ROUTER_SCHEDULER_CONFIG_PRECISE_PREFIX_CACHE = {
    "apiVersion": "inference.networking.x-k8s.io/v1alpha1",
    "kind": "EndpointPickerConfig",
    "plugins": [
        {
            "type": "prefix-cache-scorer",
            "parameters": {
                "indexerConfig": {
                    "tokenProcessorConfig": {
                        "blockSize": PREFIX_CACHE_BLOCK_SIZE,
                        "hashAlgo": PREFIX_CACHE_HASH_ALGO,
                        "hashSeed": PREFIX_CACHE_HASH_SEED,
                    }
                }
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
