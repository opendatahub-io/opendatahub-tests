from utilities.constants import MinIo, ModelAndFormat

MINIO_DATA_CONNECTION_CONFIG = {"bucket": MinIo.Buckets.EXAMPLE_MODELS}
MINIO_RUNTIME_CONFIG = {
    "runtime-name": f"{MinIo.Metadata.NAME}-ovms",
    "supported-model-formats": [{"name": ModelAndFormat.OPENVINO_IR, "version": "1"}],
    "runtime_image": MinIo.PodConfig.KSERVE_MINIO_IMAGE,
}
MINIO_INFERENCE_CONFIG = {
    "name": "loan-model",
    "model-format": ModelAndFormat.OPENVINO_IR,
    "model-version": "1",
    "model-dir": "kserve/openvino-age-gender-recognition",
}
KSERVE_MINIO_INFERENCE_CONFIG = {"model-dir": "kserve/openvino-age-gender-recognition", **MINIO_INFERENCE_CONFIG}

AGE_GENDER_INFERENCE_TYPE = "age-gender-recognition"
