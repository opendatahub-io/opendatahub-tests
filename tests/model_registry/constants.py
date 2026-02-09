from typing import Any

from ocp_resources.deployment import Deployment
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.resource import Resource
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from utilities.constants import ModelFormat


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
SECURE_MR_NAME: str = "secure-db-mr"
DB_BASE_RESOURCES_NAME: str = "db-model-registry"
DB_RESOURCE_NAME: str = f"{DB_BASE_RESOURCES_NAME}0"
MR_DB_IMAGE_DIGEST: str = (
    "public.ecr.aws/docker/library/mysql@sha256:28540698ce89bd72f985044de942d65bd99c6fadb2db105327db57f3f70564f0"
)
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
DEFAULT_MODEL_CATALOG_CM: str = "model-catalog-default-sources"
KUBERBACPROXY_STR: str = "KubeRBACProxyAvailable"
MR_POSTGRES_DB_OBJECT: dict[Any, str] = {
    Service: f"{MR_INSTANCE_NAME}-postgres",
    PersistentVolumeClaim: f"{MR_INSTANCE_NAME}-postgres-storage",
    Deployment: f"{MR_INSTANCE_NAME}-postgres",
    Secret: f"{MR_INSTANCE_NAME}-postgres-credentials",
}
MR_POSTGRES_DEPLOYMENT_NAME_STR = f"{MR_INSTANCE_NAME}-postgres"
