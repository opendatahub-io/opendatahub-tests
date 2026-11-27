from utilities.constants import MinIo, ModelAndFormat

MINIO_DATA_CONNECTION_CONFIG = {"bucket": MinIo.Buckets.EXAMPLE_MODELS}
# Same bucket, but signals `unprivileged_minio_data_connection` to create a ConnectionsAPI-typed
# Secret (`opendatahub.io/connection-type-protocol: s3`) instead of the plain static one.
MINIO_CONNECTION_DATA_CONNECTION_CONFIG = {**MINIO_DATA_CONNECTION_CONFIG, "storage-strategy": "connection"}
MINIO_RUNTIME_CONFIG = {
    "runtime-name": f"{MinIo.Metadata.NAME}-ovms",
    "supported-model-formats": [{"name": ModelAndFormat.OPENVINO_IR, "version": "1"}],
    "runtime_image": MinIo.PodConfig.KSERVE_MINIO_IMAGE,
}
MINIO_INFERENCE_CONFIG = {
    "name": "loan-model",
    "model-format": ModelAndFormat.OPENVINO_IR,
    "model-version": "1",
    "model-dir": "serving/kserve/openvino-age-gender-recognition",
}
KSERVE_MINIO_INFERENCE_CONFIG = {
    "model-dir": "serving/kserve/openvino-age-gender-recognition",
    **MINIO_INFERENCE_CONFIG,
}

AGE_GENDER_INFERENCE_TYPE = "age-gender-recognition"
