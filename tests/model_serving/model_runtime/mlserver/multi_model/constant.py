"""Constants for MLServer multi-model serving tests.

S3 path prefixes mirror ``MODEL_PATH_PREFIX`` in ``mlserver/constant.py``.
Each prefix maps to a distinct directory in the shared QE S3 bucket,
pre-staged via the standard QE population process.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# S3 path prefixes
# ---------------------------------------------------------------------------
S3_PATH_10_MODEL: str = "mlserver/model_repository"
S3_PATH_MULTI_VERSION: str = "mlserver/multi_version"
S3_PATH_ONE_FAILED: str = "mlserver/one_failed_model"
S3_PATH_MALFORMED: str = "mlserver/malformed_settings"
S3_PATH_UNSUPPORTED_IMPL: str = "mlserver/unsupported_impl"
S3_PATH_MISSING_ARTIFACT: str = "mlserver/missing_artifact"

# ---------------------------------------------------------------------------
# PVC sizes
# ---------------------------------------------------------------------------
MULTI_MODEL_PVC_SIZE: str = "10Gi"
STORAGE_EXHAUSTION_PVC_SIZE: str = "1Mi"

# ---------------------------------------------------------------------------
# Namespace names
# ---------------------------------------------------------------------------
MULTI_MODEL_E2E_NS: str = "multi-model-e2e"
MULTI_MODEL_NEG_NS: str = "multi-model-neg"
MULTI_MODEL_NFR_NS: str = "multi-model-nfr"
MULTI_MODEL_UPG_NS: str = "multi-model-upg"

# ---------------------------------------------------------------------------
# Model lists and counts
# ---------------------------------------------------------------------------
EXPECTED_MODEL_COUNT: int = 10
MODEL_NAMES: list[str] = [
    "sklearn-iris-1",
    "sklearn-iris-2",
    "xgb-classifier-1",
    "xgb-classifier-2",
    "lgbm-classifier-1",
    "lgbm-classifier-2",
    "onnx-iris-1",
    "onnx-iris-2",
    "sklearn-iris-3",
    "onnx-iris-3",
]
CONCURRENT_MODELS: list[str] = [
    "sklearn-iris-1",
    "xgb-classifier-1",
    "lgbm-classifier-1",
    "onnx-iris-1",
    "sklearn-iris-3",
]

# ---------------------------------------------------------------------------
# Shared payloads
# ---------------------------------------------------------------------------
SKLEARN_IRIS_PAYLOAD: dict[str, Any] = {
    "inputs": [
        {
            "name": "predict",
            "shape": [1, 4],
            "datatype": "FP32",
            "data": [[5.1, 3.5, 1.4, 0.2]],
        }
    ]
}

MALFORMED_PAYLOAD: dict[str, Any] = {
    "inputs": [
        {
            "name": "predict",
            "shape": [1, 99],
            "datatype": "FP32",
            "data": [[1.0]],
        }
    ]
}

# ---------------------------------------------------------------------------
# Timing constants
# ---------------------------------------------------------------------------
ISVC_READY_TIMEOUT_SEC: int = 300
STARTUP_MEASUREMENT_TIMEOUT_SEC: int = 300
POLL_INTERVAL_SEC: int = 5
MODEL_STABILITY_WAIT_SEC: int = 300
POD_SCHEDULE_TIMEOUT_SEC: int = 120
RESOURCE_EXHAUSTION_OBSERVE_SEC: int = 180
CREDENTIAL_FAILURE_TIMEOUT_SEC: int = 120

# ---------------------------------------------------------------------------
# NFR constants
# ---------------------------------------------------------------------------
FLEET_READY_TIMEOUT_SEC: int = 600
NFR_ISVC_READY_TIMEOUT_SEC: int = 600
SINGLE_MODEL_COUNT: int = 1
MULTI_MODEL_COUNT: int = 10
FLEET_SINGLE_COUNT: int = 50
FLEET_MULTI_COUNT: int = 5

V2_LOAD_PATH: str = "/v2/repository/models/{model_name}/load"
V2_UNLOAD_PATH: str = "/v2/repository/models/{model_name}/unload"
V2_INFER_PATH: str = "/v2/models/{model_name}/infer"
