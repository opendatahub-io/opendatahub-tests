from typing import Any

from utilities.constants import ModelFormat


KSERVE_CONTAINER_NAME: str = "kserve-container"
KSERVE_OVMS_SERVING_RUNTIME_PARAMS: dict[str, Any] = {
    "name": "ovms-runtime",
    "template-name": "kserve-ovms",
    "multi-model": False,
}
INFERENCE_SERVICE_PARAMS: dict[str, str] = {"name": ModelFormat.ONNX}
