"""Constants for LLMD test fixtures."""

# Common liveness probe for all multinode configurations
MULTINODE_LIVENESS_PROBE = {
    "httpGet": {"path": "/health", "port": 8000, "scheme": "HTTPS"},
    "initialDelaySeconds": 4800,
    "periodSeconds": 10,
    "timeoutSeconds": 10,
    "failureThreshold": 3,
}

# Liveness probe for single-node configurations
SINGLENODE_LIVENESS_PROBE = {
    "httpGet": {"path": "/health", "port": 8000, "scheme": "HTTPS"},
    "initialDelaySeconds": 120,
    "periodSeconds": 30,
    "timeoutSeconds": 30,
    "failureThreshold": 5,
}

# Common RoCE network annotation
ROCE_ANNOTATION = {"k8s.v1.cni.cncf.io/networks": "roce-p2"}

# Scheduler configuration for multinode prefill-decode separation
# Uses prefix-cache-scorer and load-aware-scorer for efficient distribution
MULTINODE_SCHEDULER_CONFIG_PD = """apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: pd-profile-handler
  parameters:
    threshold: 0
- type: prefill-header-handler
- type: prefill-filter
- type: decode-filter
- type: prefix-cache-scorer
- type: load-aware-scorer
- type: max-score-picker
schedulingProfiles:
- name: prefill
  plugins:
  - pluginRef: prefill-filter
  - pluginRef: prefix-cache-scorer
    weight: 2.0
  - pluginRef: load-aware-scorer
    weight: 1.0
  - pluginRef: max-score-picker
- name: decode
  plugins:
  - pluginRef: decode-filter
  - pluginRef: prefix-cache-scorer
    weight: 2.0
  - pluginRef: load-aware-scorer
    weight: 1.0
  - pluginRef: max-score-picker
"""

# Scheduler configuration for single-node prefill-decode separation
# Uses queue-scorer with prefix caching for single-node optimization
SINGLENODE_SCHEDULER_CONFIG_PD = """apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
  - type: prefill-header-handler
  - type: prefill-filter
  - type: decode-filter
  - type: max-score-picker
  - type: queue-scorer
    parameters:
      hashBlockSize: 5
      maxPrefixBlocksToMatch: 256
      lruCapacityPerServer: 31250
  - type: pd-profile-handler
    parameters:
      threshold: 0
      hashBlockSize: 5
schedulingProfiles:
  - name: prefill
    plugins:
      - pluginRef: prefill-filter
      - pluginRef: queue-scorer
        weight: 1.0
      - pluginRef: max-score-picker
  - name: decode
    plugins:
      - pluginRef: decode-filter
      - pluginRef: queue-scorer
        weight: 1.0
      - pluginRef: max-score-picker
"""
