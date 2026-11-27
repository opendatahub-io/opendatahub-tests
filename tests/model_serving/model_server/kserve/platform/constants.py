from utilities.constants import (
    KServeDeploymentType,
    ModelFormat,
    ModelInferenceRuntime,
    ModelStoragePath,
    ModelVersion,
)

KSERVE_RUNTIME_PARAMS = {
    "runtime-name": ModelInferenceRuntime.MLSERVER_RUNTIME,
    "model-format": {ModelFormat.ONNX: ModelVersion.V1},
}
SERVERLESS_ISVC_PARAMS = {
    "name": ModelFormat.ONNX,
    "model-version": ModelVersion.V1,
    "model-dir": ModelStoragePath.MNIST_ONNX,
    "deployment-mode": KServeDeploymentType.SERVERLESS,
}
