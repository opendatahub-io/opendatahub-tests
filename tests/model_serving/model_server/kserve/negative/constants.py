"""Constants for KServe negative tests."""

INVALID_S3_ACCESS_KEY: str = "NOTAVALIDACCESSKEY000000"
INVALID_S3_SIGNING_KEY: str = "invalidKeyValueNotValid0000000000000000"

CORRUPTED_MODEL_S3_PATH: str = "corrupted-model-negative-test"

KSERVE_CONTROL_PLANE_DEPLOYMENTS: tuple[str, ...] = (
    "kserve-controller-manager",
    "odh-model-controller",
)
