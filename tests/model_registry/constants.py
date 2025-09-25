from utilities.constants import Annotations


class ModelRegistryEndpoints:
    REGISTERED_MODELS: str = "/api/model_registry/v1alpha3/registered_models"


MODEL_NAME: str = "my-model"
MODEL_DESCRIPTION: str = "lorem ipsum"
DB_RESOURCES_NAME: str = "model-registry-db"
MR_INSTANCE_NAME: str = "model-registry"
MR_OPERATOR_NAME: str = "model-registry-operator"
MR_NAMESPACE: str = "rhoai-model-registries"
DEFAULT_LABEL_DICT_DB: dict[str, str] = {
    Annotations.KubernetesIo.NAME: DB_RESOURCES_NAME,
    Annotations.KubernetesIo.INSTANCE: DB_RESOURCES_NAME,
    Annotations.KubernetesIo.PART_OF: DB_RESOURCES_NAME,
}
