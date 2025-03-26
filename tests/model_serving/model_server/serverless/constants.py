from utilities.constants import KServeDeploymentType, ModelFormat, ModelVersion

ONNX_SERVERLESS_INFERENCE_SERVICE_CONFIG: dict[str, str] = {
    "name": ModelFormat.ONNX,
    "model-version": ModelVersion.OPSET13,
    "model-dir": "test-dir",
    "deployment-mode": KServeDeploymentType.SERVERLESS,
}
