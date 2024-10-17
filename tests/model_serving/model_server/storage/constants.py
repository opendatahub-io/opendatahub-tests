from typing import Any, Dict

# Models
ONNX_STR: str = "onnx"

KSERVE_CONTAINER_NAME: str = "kserve-container"
KSERVE_OVMS_SERVING_RUNTIME_PARAMS: Dict[str, Any] = {
    "name": "ovms-runtime",
    "model-name": ONNX_STR,
    "template-name": "kserve-ovms",
    "model-version": "1",
    "multi-model": False,
}
INFERENCE_SERVICE_PARAMS: Dict[str, str] = {"name": ONNX_STR}

# Storage
NFS_STR: str = "nfs"
