"""Constants for KServe canary rollout (RawDeployment) tests."""

from utilities.constants import KServeDeploymentType, ModelFormat

CANARY_FEATURE_NAME: str = "kserve-canary-rollout"
CANARY_NAMESPACE_PREFIX: str = "kserve-canary"

# Both revisions are sklearn. Traffic assertions rely on this public GCS pair:
# stable (1.0) → HTTP 200 and canary (1.3/mixedtype) → HTTP 500 for TRAFFIC_INFERENCE_INPUT.
# If Google changes or removes these artifacts, fingerprinting breaks — prefer a
# self-hosted pair long-term.
STABLE_MODEL_FORMAT: str = ModelFormat.SKLEARN
CANARY_MODEL_FORMAT: str = ModelFormat.SKLEARN

STABLE_STORAGE_URI: str = "gs://kfserving-examples/models/sklearn/1.0/model"
CANARY_STORAGE_URI: str = "gs://kfserving-examples/models/sklearn/1.3/mixedtype"

DEFAULT_DEPLOYMENT_MODE: str = KServeDeploymentType.STANDARD
DEFAULT_CANARY_TRAFFIC_PERCENT: int = 10

TRAFFIC_SAMPLE_SIZE: int = 1000
TRAFFIC_TOLERANCE_PERCENT: int = 5

# Same iris rows as the classic V1 instances payload. MLServer exposes V2 only;
# stable (1.0) → 200, canary (mixedtype) → 500 on this tensor.
_TRAFFIC_ROWS: list[list[float]] = [
    [6.8, 2.8, 4.8, 1.4],
    [6.0, 3.4, 4.5, 1.6],
]
TRAFFIC_INFERENCE_INPUT: dict = {
    "inputs": [
        {
            "name": "predict",
            "shape": [len(_TRAFFIC_ROWS), len(_TRAFFIC_ROWS[0])],
            "datatype": "FP32",
            "data": _TRAFFIC_ROWS,
        }
    ]
}
V2_INFER_PATH_TEMPLATE: str = "/v2/models/{model_name}/infer"

PROMOTION_WAIT_TIMEOUT: int = 120
