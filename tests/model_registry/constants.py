from typing import Any
from ocp_resources.resource import Resource
from utilities.constants import ModelFormat
from kubernetes.dynamic import DynamicClient
from ocp_resources.data_science_cluster import DataScienceCluster
from pytest_testconfig import config as py_config


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
MR_INSTANCE_NAME: str = "model-registry"
ISTIO_CONFIG_DICT: dict[str, Any] = {
    "authProvider": "redhat-ods-applications-auth-provider",
    "gateway": {"grpc": {"tls": {}}, "rest": {"tls": {}}},
}
DB_RESOURCES_NAME: str = "db-model-registry"
MR_DB_IMAGE_DIGEST: str = (
    "public.ecr.aws/docker/library/mysql@sha256:9de9d54fecee6253130e65154b930978b1fcc336bcc86dfd06e89b72a2588ebe"
)
MODEL_REGISTRY_DB_SECRET_STR_DATA = {
    "database-name": "model_registry",
    "database-password": "TheBlurstOfTimes",  # pragma: allowlist secret
    "database-user": "mlmduser",  # pragma: allowlist secret
}
MODEL_REGISTRY_DB_SECRET_ANNOTATIONS = {
    f"{Resource.ApiGroup.TEMPLATE_OPENSHIFT_IO}/expose-database_name": "'{.data[''database-name'']}'",
    f"{Resource.ApiGroup.TEMPLATE_OPENSHIFT_IO}/expose-password": "'{.data[''database-password'']}'",
    f"{Resource.ApiGroup.TEMPLATE_OPENSHIFT_IO}/expose-username": "'{.data[''database-user'']}'",
}


def get_mr_namespace(admin_client: DynamicClient) -> str:
    """Get the Model Registry namespace from the DSC instance.

    Args:
        admin_client: The kubernetes client

    Returns:
        The namespace for Model Registry as specified in the DSC instance

    Raises:
        ValueError: If the namespace cannot be found in the DSC instance
    """
    dsc = DataScienceCluster(client=admin_client, name=py_config["dsc_name"], ensure_exists=True)
    if not dsc.instance or not dsc.instance.spec.components.modelregistry:
        raise ValueError("ModelRegistry component not found in DSC instance")

    namespace = dsc.instance.spec.components.modelregistry.registriesNamespace
    if not namespace:
        raise ValueError("registriesNamespace not set in ModelRegistry component")

    return namespace


class LazyNamespace:
    """Lazy evaluation wrapper for MR_NAMESPACE to avoid circular imports."""

    def __init__(self) -> None:
        self._namespace = None
        self._admin_client = None

    def __str__(self) -> str:
        if self._namespace is None:
            if self._admin_client is None:
                raise ValueError("admin_client must be set before accessing MR_NAMESPACE")
            self._namespace = get_mr_namespace(admin_client=self._admin_client)
        return self._namespace

    def set_client(self, admin_client: DynamicClient) -> None:
        """Set the admin client to be used when resolving the namespace."""
        self._admin_client = admin_client


# Create a singleton instance
_MR_NAMESPACE = LazyNamespace()

# Export the singleton instance as MR_NAMESPACE
MR_NAMESPACE = _MR_NAMESPACE


# Function to initialize the namespace with a client
def init_mr_namespace(admin_client: DynamicClient) -> None:
    """Initialize MR_NAMESPACE with an admin client.

    This should be called early in test setup, typically in a session-scoped fixture.
    """
    _MR_NAMESPACE.set_client(admin_client=admin_client)
