from typing import Any

from ocp_resources.deployment import Deployment
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.resource import Resource
from ocp_resources.secret import Secret
from ocp_resources.service import Service

from tests.ai_hub.image_constants import AiHubImages
from utilities.constants import ModelCarImage, ModelFormat, RuntimeTemplates


class ModelRegistryEndpoints:
    REGISTERED_MODELS: str = "/api/model_registry/v1alpha3/registered_models"


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
MR_INSTANCE_BASE_NAME: str = "model-registry"
MR_INSTANCE_NAME: str = f"{MR_INSTANCE_BASE_NAME}0"
MR_RUNTIME_TEMPLATE: str = RuntimeTemplates.MLSERVER
SECURE_MR_NAME: str = "secure-db-mr"
DB_BASE_RESOURCES_NAME: str = "db-model-registry"
DB_RESOURCE_NAME: str = f"{DB_BASE_RESOURCES_NAME}0"
MR_DB_IMAGE_DIGEST: str = AiHubImages.MYSQL
MR_DB_MYSQL_ARGS: list[str] = ["--datadir", "/var/lib/mysql/datadir"]
# MySQL 8.4 from registry.redhat.io — supports amd64/arm64/s390x/ppc64le.
# Uses run-mysqld entrypoint which sets MYSQL_DATADIR internally, no args needed.
MR_DB_IMAGE_DIGEST_S390X: str = AiHubImages.MYSQL_S390X
MODEL_REGISTRY_DB_SECRET_STR_DATA: dict[str, str] = {
    "database-name": "model_registry",
    "database-password": "TheBlurstOfTimes",  # pragma: allowlist secret
    "database-user": "mlmduser",  # pragma: allowlist secret
}
MODEL_REGISTRY_DB_SECRET_ANNOTATIONS = {
    f"{Resource.ApiGroup.TEMPLATE_OPENSHIFT_IO}/expose-database_name": "'{.data[''database-name'']}'",
    f"{Resource.ApiGroup.TEMPLATE_OPENSHIFT_IO}/expose-password": "'{.data[''database-password'']}'",
    f"{Resource.ApiGroup.TEMPLATE_OPENSHIFT_IO}/expose-username": "'{.data[''database-user'']}'",
}

CA_CONFIGMAP_NAME = "odh-trusted-ca-bundle"
CA_MOUNT_PATH = "/etc/pki/ca-trust/extracted/pem"
CA_FILE_PATH = f"{CA_MOUNT_PATH}/ca-bundle.crt"
NUM_RESOURCES = {"num_resources": 3}
NUM_MR_INSTANCES: int = 2
MARIADB_MY_CNF = (
    "[mysqld]\nbind-address=0.0.0.0\ndefault_storage_engine=InnoDB\n"
    "binlog_format=row\ninnodb_autoinc_lock_mode=2\ninnodb_buffer_pool_size=1024M"
    "\nmax_allowed_packet=256M\n"
)
PORT_MAP = {
    "mariadb": 3306,
    "mysql": 3306,
    "postgres": 5432,
}
MODEL_REGISTRY_POD_FILTER: str = "component=model-registry"
DEFAULT_CUSTOM_MODEL_CATALOG: str = "model-catalog-sources"
SAMPLE_MODEL_NAME1 = "mistralai/Mistral-7B-Instruct-v0.3"
CUSTOM_CATALOG_ID1: str = "sample_custom_catalog1"
DEFAULT_MODEL_CATALOG_CM: str = "default-catalog-sources"
MCP_CATALOG_API_PATH: str = "/api/mcp_catalog/v1alpha1/"
AGENT_CATALOG_API_PATH: str = "/api/agent_catalog/v1alpha1/"
KUBERBACPROXY_STR: str = "KubeRBACProxyAvailable"
MR_POSTGRES_DB_OBJECT: dict[Any, str] = {
    Service: f"{MR_INSTANCE_NAME}-postgres",
    PersistentVolumeClaim: f"{MR_INSTANCE_NAME}-postgres-storage",
    Deployment: f"{MR_INSTANCE_NAME}-postgres",
    Secret: f"{MR_INSTANCE_NAME}-postgres-credentials",
}
MR_POSTGRES_DEPLOYMENT_NAME_STR = f"{MR_INSTANCE_NAME}-postgres"
CATALOG_CONTAINER: str = "catalog"
MODEL_REGISTRY_BASE_URI: str = "/api/model_registry/v1alpha3/"
MODEL_ARTIFACT: dict[str, Any] = {
    "name": "model-artifact-rest-api",
    "description": "Model artifact created via rest call",
    "uri": ModelCarImage.MLSERVER_ONNX,
    "state": "UNKNOWN",
    "modelFormatName": ModelFormat.ONNX,
    "modelFormatVersion": "v1",
    "artifactType": "model-artifact",
    "customProperties": {
        "test_ma_bool_property": {"bool_value": True, "metadataType": "MetadataBoolValue"},
        "test_ma_str_property": {"string_value": "my_value", "metadataType": "MetadataStringValue"},
    },
}

# Model Registry InferenceService defaults (MLServer / onnx)
MR_ISVC_RESOURCES: dict[str, dict[str, str]] = {
    "limits": {"cpu": "2", "memory": "4Gi"},
    "requests": {"cpu": "2", "memory": "4Gi"},
}
MR_ISVC_ARGS: list[str] = []
MR_ISVC_VOLUMES: list[dict[str, Any]] = []
MR_ISVC_VOLUME_MOUNTS: list[dict[str, str]] = []
MR_RUNTIME_CONTAINERS: dict[str, Any] = {}
MR_MODEL_SERVER_PORT: int = 8080
MR_MODEL_SERVER_URL_PATH: str = "/v2/models"
MR_ISVC_VLLM_INFERENCE: bool = False

# s390x (vLLM) overrides — applied via pytest_sessionstart when cluster_architecture == "s390x"
MR_RUNTIME_TEMPLATE_Z: str = "vllm-cpu-runtime-template"
MODEL_ARTIFACT_VLLM: dict[str, Any] = {
    "uri": "hf://TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "modelFormatName": "vLLM",
    "customProperties": {
        "HF_HUB_ENABLE_HF_TRANSFER": {"metadataType": "MetadataStringValue", "string_value": "0"},
    },
}
MR_ISVC_RESOURCES_Z: dict[str, dict[str, str]] = {
    "limits": {"cpu": "6", "memory": "12Gi"},
    "requests": {"cpu": "2", "memory": "8Gi"},
}
MR_ISVC_ARGS_Z: list[str] = ["--enforce-eager", "--max-model-len=256", "--max-num-seqs=20", "--dtype=float"]
MR_ISVC_VOLUMES_Z: list[dict[str, Any]] = [
    {"name": "shared-memory", "emptyDir": {"medium": "Memory", "sizeLimit": "32Gi"}},
    {"name": "tmp", "emptyDir": {}},
    {"name": "home", "emptyDir": {}},
]
MR_ISVC_VOLUME_MOUNTS_Z: list[dict[str, str]] = [
    {"name": "shared-memory", "mountPath": "/dev/shm"},
    {"name": "tmp", "mountPath": "/tmp"},
    {"name": "home", "mountPath": "/home/vllm"},
]
MR_RUNTIME_CONTAINERS_Z: dict[str, Any] = {
    "kserve-container": {
        "env": [
            {"name": "VLLM_CPU_KVCACHE_SPACE", "value": "4"},
            {"name": "TORCH_COMPILE_DISABLE", "value": "1"},
            {"name": "VLLM_WORKER_MULTIPROC_METHOD", "value": "spawn"},
            {"name": "OMP_NUM_THREADS", "value": "8"},
        ]
    },
}
MR_MODEL_SERVER_URL_PATH_Z: str = ""
