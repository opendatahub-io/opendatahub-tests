from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.scaled_object import ScaledObject
from kubernetes.dynamic.exceptions import ResourceNotFoundError


def get_isvc_keda_scaledobject(client: DynamicClient, isvc: InferenceService) -> list[ScaledObject]:
    """
    Get KEDA ScaledObject resources associated with an InferenceService.

    Args:
        client (DynamicClient): OCP Client to use.
        isvc (InferenceService): InferenceService object.

    Returns:
        list[ScaledObject]: A list containing the ScaledObject

    Raises:
        ResourceNotFoundError: if no ScaledObjects are found.
    """
    namespace = isvc.namespace
    scaled_object_name = isvc.name + "-predictor"

    try:
        scaled_object = ScaledObject(
            client=client,
            name=scaled_object_name,
            namespace=namespace,
            ensure_exists=True
        )
        return [scaled_object]
    except Exception as e:
        raise ResourceNotFoundError(f"{isvc.name} has no KEDA ScaledObjects: {e}")
