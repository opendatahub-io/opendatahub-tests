from utilities.constants import ModelVersion

BASE_ISVC_CONFIG: dict[str, str] = {
    "name": "isvc-replicas",
    "model-version": ModelVersion.OPSET13,
    "model-dir": "test-dir",
}

ISVC_ENV_VARS: list[dict[str, str]] = [
    {"name": "TEST_ENV_VAR1", "value": "test_value1"},
    {"name": "TEST_ENV_VAR2", "value": "test_value2"},
]

ORIGINAL_PULL_SECRET: str = "pull-secret-1"  # pragma: allowlist-secret
UPDATED_PULL_SECRET: str = "updated-pull-secret"  # pragma: allowlist-secret

# Inference logging (kserve-agent sidecar). The sink URL does not need to resolve:
# the agent posts payloads asynchronously and its readiness probe is served by the
# agent itself, so a non-existent in-cluster endpoint keeps the ISVC Ready.
ISVC_LOGGER_SINK_URL: str = "http://logger-sink.invalid.svc.cluster.local"
ISVC_LOGGER_MODE: str = "all"
