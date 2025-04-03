from utilities.constants import KServeDeploymentType, ModelFormat, ModelVersion


ONNX_SERVERLESS_BASE_INFERENCE_SERVICE_CONFIG: dict[str, str] = {
    "name": ModelFormat.ONNX,
    "model-version": ModelVersion.OPSET13,
    "deployment-mode": KServeDeploymentType.SERVERLESS,
}

ONNX_SERVERLESS_INFERENCE_SERVICE_CONFIG: dict[str, str] = {
    **ONNX_SERVERLESS_BASE_INFERENCE_SERVICE_CONFIG,
    "model-dir": "test-dir",
}
