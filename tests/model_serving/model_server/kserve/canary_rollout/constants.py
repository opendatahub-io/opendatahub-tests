"""Constants for KServe canary rollout (RawDeployment) tests."""

from tests.model_serving.model_runtime.mlserver.constant import MODEL_CONFIGS, MODEL_PATH_PREFIX
from utilities.constants import KServeDeploymentType, ModelFormat

CANARY_FEATURE_NAME: str = "kserve-canary-rollout"
CANARY_NAMESPACE_PREFIX: str = "kserve-canary"

STABLE_MODEL_FORMAT: str = ModelFormat.SKLEARN
CANARY_MODEL_FORMAT: str = ModelFormat.LIGHTGBM

STABLE_MODEL_DIR: str = f"{MODEL_PATH_PREFIX.rstrip('/')}/{STABLE_MODEL_FORMAT}"
CANARY_MODEL_DIR: str = f"{MODEL_PATH_PREFIX.rstrip('/')}/{CANARY_MODEL_FORMAT}"

DEFAULT_DEPLOYMENT_MODE: str = KServeDeploymentType.STANDARD
DEFAULT_CANARY_TRAFFIC_PERCENT: int = 10

TRAFFIC_SAMPLE_SIZE: int = 1000
TRAFFIC_TOLERANCE_PERCENT: int = 5

STABLE_INFERENCE_INPUT = MODEL_CONFIGS[STABLE_MODEL_FORMAT]["rest_query"]
CANARY_INFERENCE_INPUT = MODEL_CONFIGS[CANARY_MODEL_FORMAT]["rest_query"]

PROMOTION_WAIT_TIMEOUT: int = 120
