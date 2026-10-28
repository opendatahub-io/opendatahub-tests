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
# agent itself, so a non-existent endpoint keeps the ISVC Ready. The ".invalid"
# reserved TLD is used (IETF RFC 2606) and the trailing dot makes the hostname
# absolute, so the resolver's search list (e.g. "<name>.<ns>.svc.cluster.local")
# is never appended and the name can never resolve to a real Service.
ISVC_LOGGER_SINK_URL: str = "http://logger-sink.invalid."
ISVC_LOGGER_MODE: str = "all"
