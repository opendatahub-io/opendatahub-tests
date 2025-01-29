from typing import Any, Dict

from utilities.constants import ModelFormat


KSERVE_CONTAINER_NAME: str = "kserve-container"
KSERVE_OVMS_SERVING_RUNTIME_PARAMS: Dict[str, Any] = {
    "name": "ovms-runtime",
    "template-name": "kserve-ovms",
    "multi-model": False,
}
INFERENCE_SERVICE_PARAMS: Dict[str, str] = {"name": ModelFormat.ONNX}
