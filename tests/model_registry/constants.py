from typing import Any

from utilities.constants import ModelFormat


class ModelRegistryEndpoints:
    REGISTERED_MODELS: str = "/api/model_registry/v1alpha3/registered_models"


MR_NAMESPACE: str = "rhoai-model-registries"
MR_OPERATOR_NAME: str = "model-registry-operator"
MODEL_NAME: str = "my-model"
MODEL_DICT: dict[str, Any] = {
    "model_name": MODEL_NAME,
    "model_uri": "https://storage-place.my-company.com",
    "model_version": "2.0.0",
    "model_description": "lorem ipsum",
    "model_format": ModelFormat.ONNX,
    "model_format_version": "1",
    "model_storage_key": "my-data-connection",
    "model_storage_path": "path/to/model",
    "model_metadata": {
        "int_key": 1,
        "bool_key": False,
        "float_key": 3.14,
        "str_key": "str_value",
    },
}
MR_INSTANCE_NAME: str = "model-registry"
ISTIO_CONFIG_DICT: dict[str, Any] = {
    "authProvider": "redhat-ods-applications-auth-provider",
    "gateway": {"grpc": {"tls": {}}, "rest": {"tls": {}}},
}
DB_RESOURCES_NAME: str = "model-registry-db"
MR_DB_IMAGE_DIGEST: str = (
    "public.ecr.aws/docker/library/mysql@sha256:9de9d54fecee6253130e65154b930978b1fcc336bcc86dfd06e89b72a2588ebe"
)
