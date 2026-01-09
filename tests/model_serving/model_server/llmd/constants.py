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
PREFIX_CACHE_HASH_ALGO = "sha256"
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
