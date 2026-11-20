from utilities.constants import MinIo, ModelFormat, ModelVersion

MINIO_DATA_CONNECTION_CONFIG = {"bucket": MinIo.Buckets.EXAMPLE_MODELS}
MINIO_CONNECTION_DATA_CONNECTION_CONFIG = {"bucket": MinIo.Buckets.EXAMPLE_MODELS, "storage-strategy": "connection"}
MINIO_RUNTIME_CONFIG = {
    "runtime-name": f"{MinIo.Metadata.NAME}-mlserver",
}
# The MinIO example-models bucket has no `model.onnx` under a kserve layout; the
# modelmesh ONNX MNIST model is a single S3 object. KServe's storage initializer
# downloads the exact object to `/mnt/models/mnist.onnx`, so point MLServer at the
# file explicitly via `MLSERVER_MODEL_URI` (KServe would otherwise default it to
# the `/mnt/models` directory and fail the well-known-filename lookup).
MINIO_INFERENCE_CONFIG = {
    "name": "loan-model",
    "model-format": ModelFormat.ONNX,
    "model-version": ModelVersion.V1,
    "model-dir": "serving/modelmesh/onnx/mnist.onnx",
    "env-vars": {"MLSERVER_MODEL_URI": "/mnt/models/mnist.onnx"},
}
