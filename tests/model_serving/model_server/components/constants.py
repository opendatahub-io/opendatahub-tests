from utilities.constants import (
    KServeDeploymentType,
    ModelAndFormat,
    ModelFormat,
    ModelInferenceRuntime,
    ModelStoragePath,
    ModelVersion,
)

KSERVE_RUNTIME_PARAMS = {
    "runtime-name": ModelInferenceRuntime.OPENVINO_KSERVE_RUNTIME,
    "model-format": {ModelAndFormat.OPENVINO_IR: ModelVersion.OPSET1},
}
SERVERLESS_ISVC_PARAMS = {
    "name": ModelFormat.OPENVINO,
    "model-version": ModelVersion.OPSET1,
    "model-dir": ModelStoragePath.KSERVE_OPENVINO_EXAMPLE_MODEL,
    "deployment-mode": KServeDeploymentType.SERVERLESS,
}
