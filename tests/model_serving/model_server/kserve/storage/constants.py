from typing import Any

from utilities.constants import ModelFormat

KSERVE_OVMS_SERVING_RUNTIME_PARAMS: dict[str, Any] = {
    "name": "ovms-runtime",
    "template-name": "kserve-ovms",
    "multi-model": False,
}
INFERENCE_SERVICE_PARAMS: dict[str, str] = {"name": ModelFormat.ONNX}

# ConnectionsAPI — standalone URI CREATE test (manual 1.2). A tiny ONNX test model already used by
# the model-catalog suite (tests/ai_hub/model_catalog/constants.py).
ISVC_URI_MODEL_URI: str = "hf://dbasunag/onnx-test-model"
# Input schema for the `dbasunag/onnx-test-model` model above (confirmed via its v2 metadata
# endpoint: {"inputs": [{"name": "dense_input", "shape": [-1, 5], "datatype": "FP32"}]}). This is a
# different model than the sibling mlserver suite's own onnx test model, so it does not share that
# suite's `ONNX_REST_INPUT_QUERY` (a different input name/shape) — reusing it 400s.
ISVC_URI_ONNX_REST_INPUT_QUERY: dict[str, Any] = {
    "id": "onnx-test-model",
    "inputs": [
        {
            "name": "dense_input",
            "shape": [1, 5],
            "datatype": "FP32",
            "data": [[0.1, 0.2, 0.3, 0.4, 0.5]],
        }
    ],
}
