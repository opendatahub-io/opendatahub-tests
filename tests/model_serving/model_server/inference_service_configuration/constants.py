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
