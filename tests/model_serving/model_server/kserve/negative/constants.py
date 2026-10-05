"""Constants for KServe negative tests."""

INVALID_S3_ACCESS_KEY: str = "NOTAVALIDACCESSKEY000000"
INVALID_S3_SIGNING_KEY: str = "invalidKeyValueNotValid0000000000000000"

CORRUPTED_MODEL_S3_PATH: str = "corrupted-model-negative-test"
NONEXISTENT_STORAGE_CLASS: str = "nonexistent-sc"
WRONG_MODEL_FORMAT: str = "invalid-model-format"

KSERVE_CONTROL_PLANE_DEPLOYMENTS: tuple[str, ...] = (
    "kserve-controller-manager",
    "odh-model-controller",
)

# Boundary / edge-case constants
OVERSIZED_PAYLOAD_SIZE_BYTES: int = 6 * 1024 * 1024

# Model name boundary cases
VERY_LONG_MODEL_NAME: str = "a" * 253  # exceeds Kubernetes name limit (253 chars)
MODEL_NAME_WITH_SPECIAL_CHARS: str = "../../etc/passwd"

# Concurrent requests boundary
CONCURRENT_INVALID_REQUEST_COUNT: int = 10
